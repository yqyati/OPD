#!/usr/bin/env bash
# Completion-aware fixed-prefix-512 OPD: Qwen3-1.7B-Base <- Qwen3-4B-Instruct-2507.
# This script intentionally only defines the experiment and delegates execution
# to the shared OPD runner.  It does not submit an rjob.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export TEACHER_PREFIX_SFT_LOSS_COEF="${TEACHER_PREFIX_SFT_LOSS_COEF:-0.2}"
SFT_COEF_TAG="${TEACHER_PREFIX_SFT_LOSS_COEF/./p}"

export RUN_MODE=prefix
export PREFIX_TRAIN_DATASET="datasets/teacher_prefix/q3b_q4b_instruct2507_nothink_dapo_math17k_prefix512_from_fullresponse.parquet"
export PREFIX_TRAIN_DATASET_NAME="q3b_q4b_i2507_prefix512_completionaware"
export PREFIX_MODEL_OUTPUT_NAME_PREFIX="q3b_q4b_i2507_problem_solution_prefix512_sft${SFT_COEF_TAG}_opd_r4096_b64_n1_lr1e-5"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-1.7B-Base}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export REWARD_MODEL_INPUT_TOKENIZER="${ACTOR_MODEL_PATH}"
export STUDENT_CHAT_TEMPLATE_FILE="${STUDENT_CHAT_TEMPLATE_FILE:-${OPD_ROOT}/templates/qwen3_base_problem_solution.jinja}"

export TEACHER_PREFIX_MAX_LEN=512
export CANONICAL_EOS_TOKEN_ID=151643
export TEACHER_SOURCE_EOS_TOKEN_ID=151645
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

export ENABLE_THINKING=False
export DATA_SHUFFLE=False
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
export EVAL_OUTPUT_DIR="${OPD_ROOT}/outputs/eval/problem_solution_prefix512_eos151643"

# Tracking uses EXPERIMENT_NAME as one directory component.  Keep it short
# enough for the 255-byte filesystem component limit, while retaining the
# experiment-defining fields in the model name and output directory above.
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q3b_q4b_i2507_ps_prefix512_sft${SFT_COEF_TAG}_opd_r4096_b64_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PREFIX_TRAIN_DATASET}"
test -f "${STUDENT_CHAT_TEMPLATE_FILE}"
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"

exec bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
