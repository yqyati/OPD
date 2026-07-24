#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"
PILOT_DIR="${ROOT}/OPD/datasets/group_opd_pilot/q3b_q4b_prefix128_n4_median_vs_random_seed42"
OUTPUT_DIR="${ROOT}/OPD/outputs/interaction_mass_saturation/q3b_q4b_dapo_prefix128_p1000_seed42_$(date +%Y-%m-%d_%H-%M-%S)"

python scripts/mixed_sampling/analyze_interaction_mass_saturation.py \
  --input "${ROOT}/OPD/datasets/dapo-math-17k-teacher-aligned.parquet" \
  --candidate-dir "${PILOT_DIR}" \
  --student-model "${ROOT}/model/Qwen3-1.7B-Base" \
  --teacher-model "${ROOT}/model/Qwen3-4B-Base" \
  --output-dir "${OUTPUT_DIR}" \
  --num-prompts 1000 \
  --seed 42 \
  --gpus 0,1,2,3 \
  --batch-size 4
