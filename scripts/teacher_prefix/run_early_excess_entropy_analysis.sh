#!/usr/bin/env bash
source .env

set -euo pipefail

# Forward-only analysis using the existing matched Prefix0 and Prefix128 K=8
# rollout study. No rollout or training is repeated.
ROOT=${YANGQINGYU_ROOT}
cd "${ROOT}/OPD"

INPUT="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
PREFIX128_VALUES="${ROOT}/OPD/outputs/handoff_value/q3b_thinking_alive512_tp32_64_128_256_512_k8_p500_seed42_2026-07-17_22-03-46/per_prompt_handoff_values.jsonl"
PREFIX0_VALUES="${ROOT}/OPD/outputs/handoff_value/q3b_thinking_alive512_zero_prefix_k8_p500_seed42_2026-07-17_22-40-55/per_prompt_handoff_values.jsonl"
STUDENT="${ROOT}/model/Qwen3-1.7B-Base"
TEACHER="${ROOT}/model/Qwen3-4B-Base"
GPUS="${GPUS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTPUT="${ROOT}/OPD/outputs/teacher_prefix_entropy/q3b_q4b_early_excess_entropy_v128_minus_v0_k8_p500_seed42_$(date +%Y-%m-%d_%H-%M-%S)"

test -f "${INPUT}"
test -f "${PREFIX0_VALUES}"
test -f "${PREFIX128_VALUES}"

python scripts/teacher_prefix/analyze_early_excess_entropy.py \
  --input "${INPUT}" \
  --prefix0-values "${PREFIX0_VALUES}" \
  --prefix128-values "${PREFIX128_VALUES}" \
  --student-model "${STUDENT}" \
  --teacher-model "${TEACHER}" \
  --output-dir "${OUTPUT}" \
  --windows 1,4,8,16,32 \
  --gpus "${GPUS}" \
  --batch-size "${BATCH_SIZE}"
