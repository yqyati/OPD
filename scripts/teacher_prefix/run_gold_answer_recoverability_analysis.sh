#!/usr/bin/env bash
source .env

set -euo pipefail

# Offline analysis only. It does not modify training or rollout programs.

ROOT=${YANGQINGYU_ROOT}
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
STUDENT_MODEL="${ROOT}/model/Qwen3-1.7B-Base"
HANDOFF_DIR="${HANDOFF_DIR:-${ROOT}/OPD/outputs/handoff_value/q3b_thinking_tp1024_handoff_value_seed42_2026-07-16_22-18-00}"
OUTPUT_DIR="${ROOT}/OPD/outputs/gold_answer_recoverability/q3b_thinking_gold_answer_recoverability_seed42_$(date +%Y-%m-%d_%H-%M-%S)"
GPUS=0,1,2,3
BATCH_SIZE=32

test -f "${INPUT_DATASET}"
test -f "${STUDENT_MODEL}/config.json"
test -f "${HANDOFF_DIR}/config.json"
test -f "${HANDOFF_DIR}/per_prompt_handoff_values.jsonl"

echo "========== Gold-Answer Recoverability Analysis =========="
echo "student: ${STUDENT_MODEL}"
echo "handoff labels: ${HANDOFF_DIR}"
echo "GPUs: ${GPUS}"
echo "output: ${OUTPUT_DIR}"

python scripts/teacher_prefix/analyze_gold_answer_recoverability.py \
    --input "${INPUT_DATASET}" \
    --student-model "${STUDENT_MODEL}" \
    --handoff-dir "${HANDOFF_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --gpus "${GPUS}" \
    --batch-size "${BATCH_SIZE}"
