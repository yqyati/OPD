#!/usr/bin/env bash
source .env

set -euo pipefail

# Forward-only metric benchmark. It uses stored V_student labels only after
# scoring, and never launches new continuation rollouts.
ROOT=${YANGQINGYU_ROOT}
cd "${ROOT}/OPD"

STATE_DIR="${ROOT}/OPD/outputs/group_state_value/q3b_group_prefix128_n4_k4_p1000_seed42_2026-07-17_14-21-45"
INPUT_DATASET="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
STUDENT_MODEL="${ROOT}/model/Qwen3-1.7B-Base"
TEACHER_MODEL="${ROOT}/model/Qwen3-4B-Base"
OUTPUT_DIR="${ROOT}/OPD/outputs/group_state_metric_benchmark/q3b_q4b_group_prefix128_p1000_$(date +%Y-%m-%d_%H-%M-%S)"

GPUS=0,1,2,3
BATCH_SIZE=4

python scripts/mixed_sampling/benchmark_group_state_ranking_metrics.py \
  --input "${INPUT_DATASET}" \
  --state-values "${STATE_DIR}/group_prefix_state_values.jsonl" \
  --continuations "${STATE_DIR}/group_prefix_continuations.jsonl" \
  --student-model "${STUDENT_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpus "${GPUS}" \
  --batch-size "${BATCH_SIZE}"
