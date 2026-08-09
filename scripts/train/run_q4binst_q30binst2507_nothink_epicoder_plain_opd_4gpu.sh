#!/usr/bin/env bash
# Code plain OPD on the fixed EpiCoder 30K sample.
# Student starts from the original 4B Instruct model; no math checkpoint is
# used.  Code execution rewards are disabled: the actor uses teacher-derived
# token rm_scores through reward_manager=batch.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export RUN_MODE=plain
export ENABLE_THINKING=False
export DATA_SHUFFLE=True
export DATA_SEED=42

export PLAIN_TRAIN_DATASET="${PLAIN_TRAIN_DATASET:-datasets/epicoder-func-380k/epicoder_func_30k_seed42_verl.parquet}"
export PLAIN_TRAIN_DATASET_NAME="EpiCoder-func-380k-30K-seed42-4BInst-30BInst-NoThink-PlainOPD"
export PLAIN_MODEL_OUTPUT_NAME_PREFIX="q4binst_q30binst2507_nothink_epicoder30k_plain_opd_r4096_b96_n1_lr1e-5"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
export REWARD_MODEL_INPUT_TOKENIZER="${REWARD_MODEL_INPUT_TOKENIZER:-}"
export STUDENT_CHAT_TEMPLATE_FILE="${STUDENT_CHAT_TEMPLATE_FILE:-}"
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""

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
export TEACHER_PREFIX_SFT_LOSS_COEF=0.0
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

# 30,000 rows // batch 96 = 312 complete optimizer steps in this runner.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-312}"

# Pure token OPD: BatchRewardManager returns teacher-derived rm_scores and
# never launches a code verifier.  Do not replace this with a code reward
# manager for the main experiment.
export DISABLE_CUSTOM_REWARD_FUNCTION=True
export EXTRA_PPO_ARGS="${EXTRA_PPO_ARGS:-reward_model.reward_manager=batch}"
export SKIP_FINAL_EVAL=True
export TEST_FILE="[\"${PLAIN_TRAIN_DATASET}\"]"

export RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_epicoder30k_plain_opd_r4096_b96_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PLAIN_TRAIN_DATASET}"
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"
echo "EpiCoder code plain token-OPD"
echo "student=${ACTOR_MODEL_PATH}"
echo "teacher=${REWARD_MODEL_PATH}"
echo "dataset=${PLAIN_TRAIN_DATASET}"
echo "reward_manager=batch (no CPU code verification)"

bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
