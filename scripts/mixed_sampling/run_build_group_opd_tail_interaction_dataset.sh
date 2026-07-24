#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"

PILOT_DIR="${ROOT}/OPD/datasets/group_opd_pilot/q3b_q4b_prefix128_n8_median_vs_random_seed42"
INPUT="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
OUTPUT="${PILOT_DIR}/group_opd_high_tail32_interaction_mass.parquet"
SUMMARY="${PILOT_DIR}/high_tail32_interaction_mass_summary.json"

if [[ -f "${OUTPUT}" ]]; then
    echo "Reusing ${OUTPUT}"
    exit 0
fi

python scripts/mixed_sampling/build_group_opd_advantage_dataset.py \
  --input "${INPUT}" \
  --candidate-dir "${PILOT_DIR}" \
  --student-model "${ROOT}/model/Qwen3-1.7B-Base" \
  --teacher-model "${ROOT}/model/Qwen3-4B-Base" \
  --metric tail32_interaction_mass \
  --output "${OUTPUT}" \
  --summary "${SUMMARY}" \
  --gpus "${GPUS:-0,1}" \
  --batch-size "${BATCH_SIZE:-8}"
