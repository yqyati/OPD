#!/usr/bin/env bash
set -euo pipefail

# Offline Group OPD phase-1 value diagnosis. No training entrypoint is called.

ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
STUDENT_MODEL="${ROOT}/model/Qwen3-1.7B-Base"
NUM_PROMPTS=1000
OUTPUT_DIR="${ROOT}/OPD/outputs/group_state_value/q3b_group_prefix128_n4_k4_p${NUM_PROMPTS}_seed42_$(date +%Y-%m-%d_%H-%M-%S)"

PREFIXES_PER_PROMPT=4
PREFIX_TOKENS=128
CONTINUATIONS_PER_PREFIX=4
MAX_TOKENS=4096
TEMPERATURE=1.0
TOP_P=1.0
GPUS=0,1,2,3
REQUEST_BATCH_SIZE=32
SEED=42

test -f "${INPUT_DATASET}"
test -f "${STUDENT_MODEL}/config.json"

echo "========== Group OPD Phase-1 Student State Value Diagnosis =========="
echo "student: ${STUDENT_MODEL}"
echo "prompts: ${NUM_PROMPTS}; prefixes per prompt: ${PREFIXES_PER_PROMPT}; prefix tokens: ${PREFIX_TOKENS}"
echo "continuations per prefix: ${CONTINUATIONS_PER_PREFIX}"
echo "maximum full continuations: $((NUM_PROMPTS * PREFIXES_PER_PROMPT * CONTINUATIONS_PER_PREFIX))"
echo "GPUs: ${GPUS}"
echo "output: ${OUTPUT_DIR}"

python scripts/mixed_sampling/diagnose_group_student_state_value.py \
    --input "${INPUT_DATASET}" \
    --student-model "${STUDENT_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-prompts "${NUM_PROMPTS}" \
    --prefixes-per-prompt "${PREFIXES_PER_PROMPT}" \
    --prefix-tokens "${PREFIX_TOKENS}" \
    --continuations-per-prefix "${CONTINUATIONS_PER_PREFIX}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --gpus "${GPUS}" \
    --request-batch-size "${REQUEST_BATCH_SIZE}" \
    --seed "${SEED}"
