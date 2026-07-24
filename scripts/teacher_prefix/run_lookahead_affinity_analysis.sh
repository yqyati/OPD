#!/usr/bin/env bash
set -euo pipefail

# Offline analysis only. It does not change training or the existing diagnostics.

ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
TEACHER_MODEL="${ROOT}/model/Qwen3-4B-Base"
STUDENT_MODEL="${ROOT}/model/Qwen3-1.7B-Base"
HANDOFF_DIR="${HANDOFF_DIR:-${ROOT}/OPD/outputs/handoff_value/q3b_thinking_tp1024_handoff_value_seed42_2026-07-16_22-18-00}"
OUTPUT_DIR="${ROOT}/OPD/outputs/lookahead_affinity/q3b_4b_thinking_lookahead128_seed42_$(date +%Y-%m-%d_%H-%M-%S)"
LOOKAHEAD_TOKENS=128
GPUS=0,1,2,3,4,5,6,7
BATCH_SIZE=16

test -f "${INPUT_DATASET}"
test -f "${TEACHER_MODEL}/config.json"
test -f "${STUDENT_MODEL}/config.json"
test -f "${HANDOFF_DIR}/config.json"
test -f "${HANDOFF_DIR}/per_prompt_handoff_values.jsonl"

echo "========== Lookahead Teacher-Trajectory Affinity Analysis =========="
echo "teacher: ${TEACHER_MODEL}"
echo "student: ${STUDENT_MODEL}"
echo "handoff labels: ${HANDOFF_DIR}"
echo "lookahead tokens: ${LOOKAHEAD_TOKENS}"
echo "GPUs: ${GPUS}"
echo "output: ${OUTPUT_DIR}"

python scripts/teacher_prefix/analyze_lookahead_affinity.py \
    --input "${INPUT_DATASET}" \
    --teacher-model "${TEACHER_MODEL}" \
    --student-model "${STUDENT_MODEL}" \
    --handoff-dir "${HANDOFF_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --lookahead-tokens "${LOOKAHEAD_TOKENS}" \
    --gpus "${GPUS}" \
    --batch-size "${BATCH_SIZE}"
