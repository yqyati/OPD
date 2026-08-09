#!/usr/bin/env bash
# Generate exact 7168-token-cap no-think teacher trajectories for the
# Base-GRPO full-SFT control, using four data-parallel TP=1 workers.
set -euo pipefail
source .env
cd "${OPD_ROOT}"

INPUT_DATASET="${INPUT_DATASET:-datasets/dapo-math-17k-teacher-aligned.parquet}"
FULL_RESPONSE_DATASET="${FULL_RESPONSE_DATASET:-datasets/sft_teacher_response/q3b_q4bgrpo_nothink_dapo_math17k_full_response_7168.parquet}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen3-4B-Base-GRPO}"
test -f "${INPUT_DATASET}"
test -f "${TEACHER_MODEL}/config.json"

RESPONSE_INPUT="${INPUT_DATASET}" \
RESPONSE_OUTPUT="${FULL_RESPONSE_DATASET}" \
RESPONSE_TEACHER_MODEL="${TEACHER_MODEL}" \
RESPONSE_GPU_GROUPS="${FULL_RESPONSE_GPU_GROUPS:-0;1;2;3}" \
RESPONSE_TP=1 \
RESPONSE_MAX_TOKENS=7168 \
RESPONSE_MAX_MODEL_LEN=9216 \
RESPONSE_BATCH_SIZE="${FULL_RESPONSE_BATCH_SIZE:-64}" \
RESPONSE_TEMPERATURE=0.7 \
RESPONSE_TOP_P=0.95 \
RESPONSE_ENABLE_THINKING=False \
bash scripts/sft/run_sharded_teacher_response_generation.sh
