#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"
PILOT_DIR="${ROOT}/OPD/datasets/group_opd_pilot/q3b_q4b_prefix128_n8_median_vs_random_seed42"
OUTPUT="${PILOT_DIR}/group_opd_high_advantage.parquet"

if [[ -f "${OUTPUT}" ]]; then
  echo "Reusing Group-OPD High-Advantage dataset: ${OUTPUT}"
  exit 0
fi

python scripts/mixed_sampling/build_group_opd_advantage_dataset.py \
  --input "${ROOT}/OPD/datasets/dapo-math-17k-teacher-aligned.parquet" \
  --candidate-dir "${PILOT_DIR}" \
  --student-model "${ROOT}/model/Qwen3-1.7B-Base" \
  --teacher-model "${ROOT}/model/Qwen3-4B-Base" \
  --output "${OUTPUT}" \
  --summary "${PILOT_DIR}/high_advantage_summary.json" \
  --gpus 0,1,2,3 \
  --batch-size 4
