#!/usr/bin/env bash
source .env

set -euo pipefail

ROOT=${YANGQINGYU_ROOT}
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT="${ROOT}/OPD/datasets/dapo-math-17k-teacher-aligned.parquet"
OUTPUT_DIR="${ROOT}/OPD/datasets/group_opd_pilot/q3b_q4b_prefix128_n8_median_vs_random_seed42"

test -f "${INPUT}"
test -f "${ROOT}/model/Qwen3-1.7B-Base/config.json"
test -f "${ROOT}/model/Qwen3-4B-Base/config.json"

if [[ -f "${OUTPUT_DIR}/group_opd_high_median.parquet" && -f "${OUTPUT_DIR}/group_opd_random_state.parquet" ]]; then
    echo "Reusing matched Group-OPD pilot data: ${OUTPUT_DIR}"
    exit 0
fi

python scripts/mixed_sampling/build_group_opd_pilot_datasets.py \
  --input "${INPUT}" \
  --student-model "${ROOT}/model/Qwen3-1.7B-Base" \
  --teacher-model "${ROOT}/model/Qwen3-4B-Base" \
  --output-dir "${OUTPUT_DIR}" \
  --gpus 0,1,2,3 \
  --candidates-per-prompt 8 \
  --prefix-tokens 128 \
  --temperature 1.0 \
  --top-p 1.0 \
  --generation-batch-size 32 \
  --forward-batch-size 4 \
  --random-seed 42
