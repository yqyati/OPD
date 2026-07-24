#!/usr/bin/env bash
set -euo pipefail

# Offline only: diagnose real student continuation value for exact teacher-prefix handoffs.
# This script does not call the training entrypoint and does not modify checkpoints.

ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
STUDENT_MODEL="${ROOT}/model/Qwen3-1.7B-Base"
OUTPUT_DIR="${ROOT}/OPD/outputs/handoff_value/q3b_thinking_tp1024_handoff_value_seed42_$(date +%Y-%m-%d_%H-%M-%S)"

PREFIX_LENGTHS=0,128,256,512,1024
NUM_PROMPTS=200
CONTINUATIONS_PER_PREFIX=2
MAX_TOKENS=4096
TEMPERATURE=1.0
TOP_P=1.0
GPUS=0,1,2,3,4,5,6,7
REQUEST_BATCH_SIZE=32
SEED=42

test -f "${INPUT_DATASET}"
test -f "${STUDENT_MODEL}/config.json"

echo "========== Offline Teacher-Prefix Handoff Value Diagnosis =========="
echo "student: ${STUDENT_MODEL}"
echo "input: ${INPUT_DATASET}"
echo "candidate prefix lengths: ${PREFIX_LENGTHS}"
echo "prompts: ${NUM_PROMPTS}; continuations per prefix: ${CONTINUATIONS_PER_PREFIX}"
echo "rollout max tokens: ${MAX_TOKENS}; temperature: ${TEMPERATURE}; top_p: ${TOP_P}"
echo "GPUs: ${GPUS}"
echo "output: ${OUTPUT_DIR}"

python scripts/teacher_prefix/diagnose_handoff_value.py \
    --input "${INPUT_DATASET}" \
    --student-model "${STUDENT_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --prefix-lengths "${PREFIX_LENGTHS}" \
    --num-prompts "${NUM_PROMPTS}" \
    --continuations-per-prefix "${CONTINUATIONS_PER_PREFIX}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --gpus "${GPUS}" \
    --request-batch-size "${REQUEST_BATCH_SIZE}" \
    --seed "${SEED}"
