#!/usr/bin/env bash
source .env

set -euo pipefail

# Runs the same forward-only analysis on both clean handoff-value studies.
ROOT=${YANGQINGYU_ROOT}
cd "${ROOT}/OPD"
INPUT="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
STUDENT="${ROOT}/model/Qwen3-1.7B-Base"
TEACHER="${ROOT}/model/Qwen3-4B-Base"
GPUS="${GPUS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-8}"

run_analysis() {
  local label="$1"
  local handoff_dir="$2"
  local output="${ROOT}/OPD/outputs/teacher_handoff_interaction/${label}_$(date +%Y-%m-%d_%H-%M-%S)"
  python scripts/teacher_prefix/analyze_teacher_handoff_interaction.py \
    --input "${INPUT}" \
    --handoff-values "${handoff_dir}/per_prompt_handoff_values.jsonl" \
    --student-model "${STUDENT}" \
    --teacher-model "${TEACHER}" \
    --output-dir "${output}" \
    --gpus "${GPUS}" \
    --batch-size "${BATCH_SIZE}"
}

run_analysis \
  q3b_tp32_64_128_k8 \
  "${ROOT}/OPD/outputs/handoff_value/q3b_thinking_tp32_64_128_value_k8_seed42_2026-07-17_14-00-35"
run_analysis \
  q3b_alive512_tp128_256_512_k8 \
  "${ROOT}/OPD/outputs/handoff_value/q3b_thinking_alive512_tp128_256_512_k8_p500_seed42_2026-07-17_15-47-26"
