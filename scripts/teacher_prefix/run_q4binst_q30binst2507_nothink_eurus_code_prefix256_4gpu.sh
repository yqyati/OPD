#!/usr/bin/env bash
# Build one reusable, exact-token 7k teacher-trajectory asset for Eurus-Code,
# then slice a completion-aware fixed 256-token prefix dataset from it.
# Four independent vLLM workers are used (one H200 / TP=1 per worker).
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export RESPONSE_INPUT="datasets/eurus-2-code-verl/data/train-00000.parquet"
export RESPONSE_OUTPUT="datasets/sft_teacher_response/q4binst_q30binst2507_nothink_eurus_code_full_response_7168.parquet"
export RESPONSE_TEACHER_MODEL="${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507"
export RESPONSE_GPU_GROUPS="0;1;2;3"
export RESPONSE_TP=1
export RESPONSE_MAX_TOKENS=7168
export RESPONSE_MAX_MODEL_LEN=10240
export RESPONSE_BATCH_SIZE="${RESPONSE_BATCH_SIZE:-64}"
export RESPONSE_TEMPERATURE=0.7
export RESPONSE_TOP_P=0.95
export RESPONSE_ENABLE_THINKING=False

test -f "${RESPONSE_INPUT}"
test -f "${RESPONSE_TEACHER_MODEL}/config.json"

echo "[teacher rollout] Eurus-Code: four independent TP=1 workers; batch=${RESPONSE_BATCH_SIZE}/GPU"
bash scripts/sft/run_sharded_teacher_response_generation.sh

PREFIX_DATASET="datasets/teacher_prefix/q4binst_q30binst2507_nothink_eurus_code_prefix256.parquet"
# Rebuild is intentionally cheap (pure local slicing from the saved trajectory)
# and replaces the first Pandas-written artifact, which is unreadable by this
# environment's HF Dataset parquet loader.  It never invokes teacher rollout.
echo "[prefix build] completion-aware fixed prefix=256 from saved 7k teacher trajectories"
python scripts/teacher_prefix/build_prefix_dataset_from_teacher_responses.py \
    --input "${RESPONSE_OUTPUT}" \
    --source "${RESPONSE_INPUT}" \
    --output "${PREFIX_DATASET}" \
    --prefix-length 256 \
    --force

python - "${RESPONSE_INPUT}" "${RESPONSE_OUTPUT}" "${PREFIX_DATASET}" <<'PY'
import sys
import pyarrow.parquet as pq

source_path, response_path, prefix_path = sys.argv[1:]
source_rows = pq.ParquetFile(source_path).metadata.num_rows
responses = pq.read_table(
    response_path,
    columns=["teacher_response_enable_thinking"],
)
prefix = pq.read_table(
    prefix_path,
    columns=["teacher_prefix_token_len", "teacher_prefix_finish_reason"],
)
response_rows = len(responses)
prefix_rows = len(prefix)
thinking = responses["teacher_response_enable_thinking"].to_pylist()
prefix_lengths = prefix["teacher_prefix_token_len"].to_pylist()
prefix_reasons = prefix["teacher_prefix_finish_reason"].to_pylist()

if source_rows != 25110:
    raise RuntimeError(f"Unexpected Eurus-Code row count: {source_rows} (expected 25110)")
if response_rows != source_rows or prefix_rows != source_rows:
    raise RuntimeError("Source, full-response, and prefix row counts must match")
if any(bool(value) for value in thinking):
    raise RuntimeError("Expected no-think teacher trajectories")
if not all(1 <= int(length) <= 256 for length in prefix_lengths):
    raise RuntimeError("Invalid prefix token lengths")
stop_count = sum(reason == "stop" for reason in prefix_reasons)
length_indices = [idx for idx, reason in enumerate(prefix_reasons) if reason == "length"]
if stop_count + len(length_indices) != prefix_rows or any(prefix_lengths[idx] != 256 for idx in length_indices):
    raise RuntimeError("Invalid completion-aware prefix boundary")
print(
    "Validated Eurus-Code prefix256: "
    f"rows={prefix_rows}, suffix_opd_rows={len(length_indices)}, "
    f"complete_sft_only_rows={stop_count}"
)
PY
