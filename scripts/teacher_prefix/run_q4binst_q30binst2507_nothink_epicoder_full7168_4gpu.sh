#!/usr/bin/env bash
# Generate one reusable exact-token 7k teacher-response asset for EpiCoder 30K.
# Four independent TP=1 workers are used; this stage is reused by later prefix
# experiments and must not be rerun merely to change the prefix length.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export RESPONSE_INPUT="${RESPONSE_INPUT:-datasets/epicoder-func-380k/epicoder_func_30k_seed42_verl.parquet}"
export RESPONSE_OUTPUT="${RESPONSE_OUTPUT:-datasets/sft_teacher_response/q4binst_q30binst2507_nothink_epicoder30k_full_response_7168.parquet}"
export RESPONSE_TEACHER_MODEL="${RESPONSE_TEACHER_MODEL:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
export RESPONSE_GPU_GROUPS="${RESPONSE_GPU_GROUPS:-0;1;2;3}"
export RESPONSE_TP=1
export RESPONSE_MAX_TOKENS=7168
export RESPONSE_MAX_MODEL_LEN=10240
export RESPONSE_BATCH_SIZE="${RESPONSE_BATCH_SIZE:-64}"
export RESPONSE_TEMPERATURE=0.7
export RESPONSE_TOP_P=0.95
export RESPONSE_ENABLE_THINKING=False

test -f "${RESPONSE_INPUT}"
test -f "${RESPONSE_TEACHER_MODEL}/config.json"
echo "[teacher rollout] EpiCoder 30K, full response cap 7168; four TP=1 workers"
bash scripts/sft/run_sharded_teacher_response_generation.sh

/root/miniconda3/envs/verl/bin/python - "${RESPONSE_INPUT}" "${RESPONSE_OUTPUT}" <<'PY'
import sys
import pyarrow.parquet as pq

source, response = sys.argv[1:]
source_rows = pq.ParquetFile(source).metadata.num_rows
table = pq.read_table(response, columns=[
    "teacher_response_enable_thinking",
    "teacher_response_token_len",
    "teacher_response_finish_reason",
])
if len(table) != source_rows or source_rows != 30000:
    raise RuntimeError(f"row mismatch: source={source_rows}, responses={len(table)}")
if any(bool(x) for x in table["teacher_response_enable_thinking"].to_pylist()):
    raise RuntimeError("teacher response asset contains thinking-enabled rows")
lengths = table["teacher_response_token_len"].to_pylist()
if any(int(x) < 1 or int(x) > 7168 for x in lengths):
    raise RuntimeError("invalid teacher response token length")
reasons = table["teacher_response_finish_reason"].to_pylist()
print(f"validated EpiCoder teacher asset: rows={len(table)}, stop={reasons.count('stop')}, length={reasons.count('length')}")
PY
