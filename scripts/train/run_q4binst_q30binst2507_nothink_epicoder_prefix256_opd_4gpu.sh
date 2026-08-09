#!/usr/bin/env bash
# Completion-aware prefix-256 code OPD on the fixed EpiCoder 30K sample.
# The prefix parquet must be built from the already saved full teacher
# trajectories; this runner never re-rolls out the teacher.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export RUN_MODE=prefix
export ENABLE_THINKING=False
export DATA_SHUFFLE=True
export DATA_SEED=42

export PREFIX_TRAIN_DATASET="${PREFIX_TRAIN_DATASET:-datasets/teacher_prefix/q4binst_q30binst2507_nothink_epicoder30k_prefix256.parquet}"
export PREFIX_TRAIN_DATASET_NAME="EpiCoder-func-380k-30K-seed42-4BInst-30BInst-NoThink-Prefix256-CompletionAware"
export TEACHER_PREFIX_SFT_LOSS_COEF="${TEACHER_PREFIX_SFT_LOSS_COEF:-0.1}"
SFT_COEF_TAG="${TEACHER_PREFIX_SFT_LOSS_COEF/./p}"
export PREFIX_MODEL_OUTPUT_NAME_PREFIX="q4binst_q30binst2507_nothink_epicoder30k_prefix256_sft${SFT_COEF_TAG}_opd_r4096_b96_n1_lr1e-5"

# Prefix OPD is an independent variant and starts from the original student,
# not from the full-response SFT checkpoint.
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
export REWARD_MODEL_INPUT_TOKENIZER=""
export STUDENT_CHAT_TEMPLATE_FILE=""
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""
export TEACHER_PREFIX_MAX_LEN=256
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

# 2048 native-prompt tokens plus the fixed 256-token teacher prefix.
export MAX_PROMPT_LENGTH=2304
export MAX_RESP_LENGTH=4096
export MAX_VAL_RESP_LENGTH=4096
export MINI_BATCH_SIZE=96
export N_RESPONSES=1
export LR="${LR:-1e-5}"
export TOTAL_EPOCHS=1
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-312}"
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p

# Code OPD uses only teacher-token rm_scores.  No CPU program execution or
# true-reward verifier is permitted in this experiment.
export DISABLE_CUSTOM_REWARD_FUNCTION=True
export EXTRA_PPO_ARGS="reward_model.reward_manager=batch"
# The common runner's final evaluator is math-only.  rjob_launch.bash invokes
# the code evaluator explicitly after the model is merged.
export SKIP_FINAL_EVAL=True
export TEST_FILE="[\"${PREFIX_TRAIN_DATASET}\"]"

export RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_epicoder30k_prefix256_sft${SFT_COEF_TAG}_opd_r4096_b96_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PREFIX_TRAIN_DATASET}"
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"
echo "EpiCoder prefix256 token-OPD: completion-aware exact teacher token IDs; reward_manager=batch; no CPU code verification"

exec bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
