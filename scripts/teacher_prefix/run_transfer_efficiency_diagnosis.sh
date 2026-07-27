#!/usr/bin/env bash
source .env

set -euo pipefail

# Offline only: measure teacher-to-student transfer efficiency at prefix handoffs.
# Reuses an existing student handoff diagnosis and never calls the training entrypoint.

ROOT=${YANGQINGYU_ROOT}
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
TEACHER_MODEL="${ROOT}/model/Qwen3-4B-Base"
STUDENT_HANDOFF_DIR="${STUDENT_HANDOFF_DIR:-${ROOT}/OPD/outputs/handoff_value/q3b_thinking_tp1024_handoff_value_seed42_2026-07-16_22-18-00}"
OUTPUT_DIR="${ROOT}/OPD/outputs/transfer_efficiency/q3b_4b_thinking_prefix_transfer_seed42_$(date +%Y-%m-%d_%H-%M-%S)"
GPUS=0,1,2,3,4,5,6,7
REQUEST_BATCH_SIZE=32

test -f "${INPUT_DATASET}"
test -f "${TEACHER_MODEL}/config.json"
test -f "${STUDENT_HANDOFF_DIR}/handoff_rollouts.jsonl"
test -f "${STUDENT_HANDOFF_DIR}/config.json"

echo "========== Offline Teacher-to-Student Transfer Efficiency =========="
echo "teacher: ${TEACHER_MODEL}"
echo "input: ${INPUT_DATASET}"
echo "reused student diagnosis: ${STUDENT_HANDOFF_DIR}"
echo "GPUs: ${GPUS}"
echo "output: ${OUTPUT_DIR}"

python scripts/teacher_prefix/diagnose_transfer_efficiency.py \
    --input "${INPUT_DATASET}" \
    --teacher-model "${TEACHER_MODEL}" \
    --student-handoff-dir "${STUDENT_HANDOFF_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --gpus "${GPUS}" \
    --request-batch-size "${REQUEST_BATCH_SIZE}"
