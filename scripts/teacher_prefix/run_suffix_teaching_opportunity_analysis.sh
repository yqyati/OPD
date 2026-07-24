#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"
HANDOFF_DIR="${ROOT}/OPD/outputs/handoff_value/q3b_thinking_alive512_tp128_256_512_k8_p500_seed42_2026-07-17_15-47-26"
OUTPUT_DIR="${ROOT}/OPD/outputs/suffix_teaching_opportunity/q3b_alive512_tp128_256_512_probe128_$(date +%Y-%m-%d_%H-%M-%S)"

python scripts/teacher_prefix/analyze_suffix_teaching_opportunity.py \
  --input "${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet" \
  --handoff-rollouts "${HANDOFF_DIR}/handoff_rollouts.jsonl" \
  --student-model "${ROOT}/model/Qwen3-1.7B-Base" \
  --teacher-model "${ROOT}/model/Qwen3-4B-Base" \
  --output-dir "${OUTPUT_DIR}" \
  --probe-tokens 128 \
  --gpus 0,1 \
  --batch-size 8
