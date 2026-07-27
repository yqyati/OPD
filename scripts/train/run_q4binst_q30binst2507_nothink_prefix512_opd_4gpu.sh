#!/usr/bin/env bash
# Completion-aware fixed-prefix-512 OPD:
# Qwen3-4B-Instruct-2507 <- Qwen3-30B-A3B-Instruct-2507, native no-think.
# This script trains, merges, and evaluates locally; it never submits an rjob.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

# Match the completed 4B-Instruct <- 30B-Instruct plain-OPD control.  The only
# method change is a fixed 512-token teacher prefix with prefix-token CE.
export RUN_MODE=prefix
export PREFIX_TRAIN_DATASET="${PREFIX_TRAIN_DATASET:-datasets/teacher_prefix/q4binst_q30binst2507_nothink_dapo_math17k_prefix512.parquet}"
export PREFIX_TRAIN_DATASET_NAME="q4binst_q30binst2507_nothink_prefix512_completionaware"
export TEACHER_PREFIX_SFT_LOSS_COEF="${TEACHER_PREFIX_SFT_LOSS_COEF:-0.1}"
SFT_COEF_TAG="${TEACHER_PREFIX_SFT_LOSS_COEF/./p}"
export PREFIX_MODEL_OUTPUT_NAME_PREFIX="q4binst_q30binst2507_nothink_prefix512_sft${SFT_COEF_TAG}_opd_r4096_b96_n1_lr1e-5"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
# Student and teacher use the same native Qwen3-Instruct tokenizer/template/EOS
# contract.  Leave all Base<->Instruct bridge settings disabled.
export REWARD_MODEL_INPUT_TOKENIZER="${REWARD_MODEL_INPUT_TOKENIZER:-}"
export STUDENT_CHAT_TEMPLATE_FILE="${STUDENT_CHAT_TEMPLATE_FILE:-}"
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""
export TEACHER_PREFIX_MAX_LEN=512
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

export ENABLE_THINKING=False
export DATA_SHUFFLE=True
export DATA_SEED=42
export MAX_PROMPT_LENGTH=2048
export MAX_RESP_LENGTH=4096
export MAX_VAL_RESP_LENGTH=4096
export MINI_BATCH_SIZE=96
export N_RESPONSES=1
export LR="${LR:-1e-5}"
export TOTAL_EPOCHS=1
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
# Keep the historical Instruct<-Instruct plain-OPD evaluation protocol.
export EVAL_MAX_TOKENS=9192
export EVAL_OUTPUT_DIR="${OPD_ROOT}/outputs/eval/q4binst_q30binst2507_nothink_prefix512"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_prefix512_sft${SFT_COEF_TAG}_opd_r4096_b96_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PREFIX_TRAIN_DATASET}" || {
    echo "Missing prefix dataset: ${PREFIX_TRAIN_DATASET}" >&2
    echo "Build it from the saved 7k teacher trajectories before training." >&2
    exit 1
}
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"

exec bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
