#!/usr/bin/env bash
# Build a reusable 7k full-response teacher asset, then slice its exact
# fixed-128 teacher-prefix dataset for Qwen3-4B-Instruct-2507 <-
# Qwen3-30B-A3B-Instruct-2507.
#
# This is a data-preparation script only. It does not submit an rjob and does
# not start PPO/Ray. vLLM uses its normal EOS behavior: a response that reaches
# <|im_end|> before 7k stops there and is never continued. Later prefix lengths
# are sliced from the exact saved token IDs without re-sampling the teacher.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${INPUT_DATASET:-datasets/dapo-math-17k-teacher-aligned.parquet}"
FULL_RESPONSE_DATASET="${FULL_RESPONSE_DATASET:-datasets/sft_teacher_response/q4binst_q30binst2507_nothink_full_response_7168.parquet}"
OUTPUT_DATASET="${OUTPUT_DATASET:-datasets/teacher_prefix/q4binst_q30binst2507_nothink_dapo_math17k_prefix128.parquet}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
FULL_RESPONSE_GPU_GROUPS="${FULL_RESPONSE_GPU_GROUPS:-0;1;2;3}"

test -f "${INPUT_DATASET}"
test -f "${TEACHER_MODEL}/config.json"

# Four data-parallel vLLM workers, one per GPU.  This is deliberately not a
# single TP=4 worker: each worker owns one input shard and writes an independent
# resumable response_shard_XX jsonl/parquet before the parent merges them.
# On the four H200s requested for this run, one 30B-A3B BF16 replica plus a
# 64-sequence / 7k-generation KV cache fits with useful headroom.  This keeps
# all four data-parallel rollout lanes well utilized.
RESPONSE_INPUT="${INPUT_DATASET}" \
RESPONSE_OUTPUT="${FULL_RESPONSE_DATASET}" \
RESPONSE_TEACHER_MODEL="${TEACHER_MODEL}" \
RESPONSE_GPU_GROUPS="${FULL_RESPONSE_GPU_GROUPS}" \
RESPONSE_TP=1 \
RESPONSE_MAX_TOKENS=7168 \
RESPONSE_MAX_MODEL_LEN=9216 \
RESPONSE_BATCH_SIZE="${FULL_RESPONSE_BATCH_SIZE:-64}" \
RESPONSE_TEMPERATURE=0.7 \
RESPONSE_TOP_P=0.95 \
RESPONSE_ENABLE_THINKING=False \
bash scripts/sft/run_sharded_teacher_response_generation.sh

if [ ! -f "${OUTPUT_DATASET}" ]; then
    python scripts/teacher_prefix/build_prefix_dataset_from_teacher_responses.py \
        --input "${FULL_RESPONSE_DATASET}" \
        --output "${OUTPUT_DATASET}" \
        --prefix-length 128
else
    echo "Reusing existing fixed-prefix dataset: ${OUTPUT_DATASET}"
fi

python - "${OUTPUT_DATASET}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_parquet(path)
required = {
    "teacher_prefix_token_ids",
    "teacher_prefix_token_len",
    "teacher_prefix_finish_reason",
    "teacher_prefix_model",
    "teacher_prefix_enable_thinking",
}
missing = required.difference(df.columns)
if missing:
    raise RuntimeError(f"Missing prefix columns: {sorted(missing)}")
if len(df) != 17917:
    raise RuntimeError(f"Unexpected row count {len(df)}; expected 17917")
if (df.teacher_prefix_token_len > 128).any() or (df.teacher_prefix_token_len <= 0).any():
    raise RuntimeError("Invalid teacher prefix lengths")
if df.teacher_prefix_enable_thinking.astype(bool).any():
    raise RuntimeError("Prefix data unexpectedly has thinking enabled")
stopped = df.teacher_prefix_finish_reason.eq("stop")
rollout = df.teacher_prefix_finish_reason.eq("length")
if (~(stopped | rollout)).any():
    raise RuntimeError("Unexpected teacher prefix finish reason")
if not (df.loc[rollout, "teacher_prefix_token_len"] == 128).all():
    raise RuntimeError("All suffix-OPD rows must have exactly 128 prefix tokens")
print(
    f"Validated {path}: rows={len(df)}, "
    f"suffix_opd_rows={int(rollout.sum())}, "
    f"short_complete_sft_only_rows={int(stopped.sum())}"
)
PY
