#!/usr/bin/env bash
# Linear teacher-scaffold curriculum OPD:
# Qwen3-4B-Instruct-2507 <- Qwen3-30B-A3B-Instruct-2507, native no-think.
# At PPO step 1 every live trajectory receives teacher_prefix[:1024]; the
# boundary decreases linearly every step and is exactly zero at the last step.
# This is one uninterrupted PPO run, not a sequence of resumed fixed-prefix
# jobs.  The final step is ordinary plain OPD, while the student has been
# progressively exposed to less teacher-provided state throughout training.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export RUN_MODE=prefix
export PREFIX_TRAIN_DATASET="${PREFIX_TRAIN_DATASET:-datasets/teacher_prefix/q4binst_q30binst2507_nothink_dapo_math17k_prefix1024.parquet}"
export PREFIX_TRAIN_DATASET_NAME="q4binst_q30binst2507_nothink_prefix1024_to0_linear_completionaware"
export TEACHER_PREFIX_SFT_LOSS_COEF="${TEACHER_PREFIX_SFT_LOSS_COEF:-0.1}"
SFT_COEF_TAG="${TEACHER_PREFIX_SFT_LOSS_COEF/./p}"
export PREFIX_MODEL_OUTPUT_NAME_PREFIX="q4binst_q30binst2507_nothink_linear_prefix1024_to0_sft${SFT_COEF_TAG}_opd_r4096_b96_n1_lr1e-5"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
export REWARD_MODEL_INPUT_TOKENIZER="${REWARD_MODEL_INPUT_TOKENIZER:-}"
export STUDENT_CHAT_TEMPLATE_FILE="${STUDENT_CHAT_TEMPLATE_FILE:-}"
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""
export TEACHER_PREFIX_MAX_LEN=1024
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

export ENABLE_THINKING=False
export DATA_SHUFFLE=True
export DATA_SEED=42
# The original math prompts were admitted under a 2048-token limit.  Reserve
# another 1024 positions for the initial scaffold so none disappear merely
# because this curriculum starts long.
export MAX_PROMPT_LENGTH=3072
export MAX_RESP_LENGTH=4096
export MAX_VAL_RESP_LENGTH=4096
export MINI_BATCH_SIZE=96
export N_RESPONSES=1
export LR="${LR:-1e-5}"
export TOTAL_EPOCHS=1
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
export EVAL_MAX_TOKENS=9192
export EVAL_OUTPUT_DIR="${OPD_ROOT}/outputs/eval/q4binst_q30binst2507_nothink_linear_prefix1024_to0"

# `linear_prefix_curriculum` is score-free: it slices the cached 1024-token
# prefixes directly in the trainer, adds no teacher forward pass, and logs
# online_prefix/curriculum_target_len plus the selected-length statistics.
export EXTRA_PPO_ARGS="${EXTRA_PPO_ARGS:-actor_rollout_ref.rollout.online_prefix_selection.enable=True actor_rollout_ref.rollout.online_prefix_selection.selection_rule=linear_prefix_curriculum actor_rollout_ref.rollout.online_prefix_selection.curriculum_start_len=1024 actor_rollout_ref.rollout.online_prefix_selection.curriculum_end_len=0}"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_linear_prefix1024_to0_sft${SFT_COEF_TAG}_opd_r4096_b96_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PREFIX_TRAIN_DATASET}" || {
    echo "Missing prefix1024 dataset: ${PREFIX_TRAIN_DATASET}" >&2
    echo "Build it with scripts/teacher_prefix/run_q4binst_q30binst2507_nothink_prefix1024_4gpu.sh" >&2
    exit 1
}
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"

exec bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
