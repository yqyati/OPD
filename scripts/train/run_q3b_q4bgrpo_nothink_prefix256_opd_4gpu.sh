#!/usr/bin/env bash
# Fixed prefix=256 SFT + suffix OPD, math/no-think:
# Qwen3-1.7B-Base <- Qwen3-4B-Base-GRPO.
# This script runs only inside an existing four-GPU allocation and never
# submits an rjob by itself.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export RUN_MODE=prefix
PREFIX_LENGTH="${PREFIX_LENGTH:-256}"
export PREFIX_TRAIN_DATASET="${PREFIX_TRAIN_DATASET:-datasets/teacher_prefix/q3b_q4bgrpo_nothink_dapo_math17k_prefix${PREFIX_LENGTH}.parquet}"
export PREFIX_TRAIN_DATASET_NAME="q3b_q4bgrpo_nothink_prefix${PREFIX_LENGTH}_completionaware"
export TEACHER_PREFIX_SFT_LOSS_COEF="${TEACHER_PREFIX_SFT_LOSS_COEF:-0.1}"
SFT_COEF_TAG="${TEACHER_PREFIX_SFT_LOSS_COEF/./p}"
export PREFIX_MODEL_OUTPUT_NAME_PREFIX="q3b_q4bgrpo_nothink_prefix${PREFIX_LENGTH}_sft${SFT_COEF_TAG}_opd_r4096_b64_n1_lr1e-5"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-1.7B-Base}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Base-GRPO}"
# Empty means use each model's official tokenizer-provided ChatML template.
export STUDENT_CHAT_TEMPLATE_FILE=""
export REWARD_MODEL_INPUT_TOKENIZER="${REWARD_MODEL_INPUT_TOKENIZER:-}"
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""
export TEACHER_PREFIX_MAX_LEN="${PREFIX_LENGTH}"
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

export ENABLE_THINKING=False
export DATA_SHUFFLE=True
export DATA_SEED=42
export MAX_PROMPT_LENGTH=2048
export MAX_RESP_LENGTH=4096
export MAX_VAL_RESP_LENGTH=4096
export MINI_BATCH_SIZE=64
export N_RESPONSES=1
export LR="${LR:-1e-5}"
export TOTAL_EPOCHS=1
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
export EVAL_MAX_TOKENS=7168
export EVAL_OUTPUT_DIR="${OPD_ROOT}/outputs/eval/q3b_q4bgrpo_nothink_prefix${PREFIX_LENGTH}"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q3b_q4bgrpo_nothink_prefix${PREFIX_LENGTH}_sft${SFT_COEF_TAG}_opd_r4096_b64_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PREFIX_TRAIN_DATASET}" || {
    echo "Missing prefix dataset: ${PREFIX_TRAIN_DATASET}" >&2
    exit 1
}
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"

exec bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
