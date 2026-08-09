#!/usr/bin/env bash
# Build the completion-aware fixed-prefix=256 training data for
# Qwen3-1.7B-Base <- Qwen3-4B-Base-GRPO, math/no-think.
#
# This is deliberately a four-way data-parallel teacher rollout: one TP=1
# vLLM worker per GPU.  Do not replace it with a TP=4 worker.  The first stage
# persists the exact complete sampled teacher token trajectories (up to 4096
# new tokens); the second stage slices them without re-sampling.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

PREFIX_LENGTH="${PREFIX_LENGTH:-256}"
INPUT_DATASET="${INPUT_DATASET:-datasets/dapo-math-17k-teacher-aligned.parquet}"
FULL_RESPONSE_DATASET="${FULL_RESPONSE_DATASET:-datasets/sft_teacher_response/q3b_q4bgrpo_nothink_dapo_math17k_full_response_4096.parquet}"
OUTPUT_DATASET="${OUTPUT_DATASET:-datasets/teacher_prefix/q3b_q4bgrpo_nothink_dapo_math17k_prefix${PREFIX_LENGTH}.parquet}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen3-4B-Base-GRPO}"

test -f "${INPUT_DATASET}"
test -f "${TEACHER_MODEL}/config.json"

# The sharded launcher is resumable per worker and merges rows back into the
# original order.  `0;1;2;3` means four independent TP=1 model replicas.
RESPONSE_INPUT="${INPUT_DATASET}" \
RESPONSE_OUTPUT="${FULL_RESPONSE_DATASET}" \
RESPONSE_TEACHER_MODEL="${TEACHER_MODEL}" \
RESPONSE_GPU_GROUPS="${FULL_RESPONSE_GPU_GROUPS:-0;1;2;3}" \
RESPONSE_TP=1 \
RESPONSE_MAX_TOKENS=4096 \
RESPONSE_MAX_MODEL_LEN=6144 \
RESPONSE_BATCH_SIZE="${FULL_RESPONSE_BATCH_SIZE:-128}" \
RESPONSE_TEMPERATURE=0.7 \
RESPONSE_TOP_P=0.95 \
RESPONSE_ENABLE_THINKING=False \
bash scripts/sft/run_sharded_teacher_response_generation.sh

if [ ! -f "${OUTPUT_DATASET}" ]; then
    python scripts/teacher_prefix/build_prefix_dataset_from_teacher_responses.py \
        --input "${FULL_RESPONSE_DATASET}" \
        --source "${INPUT_DATASET}" \
        --output "${OUTPUT_DATASET}" \
        --prefix-length "${PREFIX_LENGTH}"
else
    echo "Reusing existing fixed-prefix dataset: ${OUTPUT_DATASET}"
fi

python - "${INPUT_DATASET}" "${FULL_RESPONSE_DATASET}" "${OUTPUT_DATASET}" "${PREFIX_LENGTH}" <<'PY'
import sys
import pandas as pd

source_path, response_path, prefix_path, prefix_length_arg = sys.argv[1:]
prefix_length = int(prefix_length_arg)
source = pd.read_parquet(source_path)
responses = pd.read_parquet(response_path)
prefix = pd.read_parquet(prefix_path)

if len(source) != len(responses) or len(source) != len(prefix):
    raise RuntimeError(
        f"row mismatch: source={len(source)}, responses={len(responses)}, prefix={len(prefix)}"
    )
required_response = {
    "teacher_response_token_ids", "teacher_response_finish_reason",
    "teacher_response_enable_thinking",
}
required_prefix = {
    "teacher_prefix_token_ids", "teacher_prefix_token_len",
    "teacher_prefix_finish_reason", "teacher_prefix_enable_thinking",
}
for label, frame, required in (
    ("full responses", responses, required_response),
    ("prefix data", prefix, required_prefix),
):
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{label} missing columns: {sorted(missing)}")
if responses["teacher_response_enable_thinking"].astype(bool).any():
    raise RuntimeError("teacher full responses unexpectedly have thinking enabled")
if prefix["teacher_prefix_enable_thinking"].astype(bool).any():
    raise RuntimeError("teacher prefix data unexpectedly has thinking enabled")
if (prefix["teacher_prefix_token_len"] <= 0).any() or (prefix["teacher_prefix_token_len"] > prefix_length).any():
    raise RuntimeError("invalid prefix lengths")
stopped = prefix["teacher_prefix_finish_reason"].eq("stop")
rollout = prefix["teacher_prefix_finish_reason"].eq("length")
if (~(stopped | rollout)).any():
    raise RuntimeError("unexpected prefix finish reason")
if not (prefix.loc[rollout, "teacher_prefix_token_len"] == prefix_length).all():
    raise RuntimeError(f"all suffix-OPD examples must contain exactly {prefix_length} prefix tokens")
print(
    f"Validated {prefix_path}: rows={len(prefix)}, "
    f"suffix_opd_rows={int(rollout.sum())}, "
    f"short_complete_sft_only_rows={int(stopped.sum())}, prefix_length={prefix_length}"
)
PY
