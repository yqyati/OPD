#!/usr/bin/env bash
# Plain science OPD: Qwen3-4B-Instruct-2507 <- Qwen3-30B-A3B-Instruct-2507.
# The true MCQ/numeric verifier is logged only.  The optimization reward stays
# the standard teacher-derived token rm_scores used by plain OPD.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export RUN_MODE=plain
export PLAIN_TRAIN_DATASET="${PLAIN_TRAIN_DATASET:-datasets/openscience_reasoning2/openscience_reasoning2_science25k_mc18p75k_numeric6p25k_seed42_verl.parquet}"
export PLAIN_TRAIN_DATASET_NAME="OpenScienceReasoning2-Science25K-4BInst-30BInst-NoThink-PlainOPD"
export PLAIN_MODEL_OUTPUT_NAME_PREFIX="${PLAIN_MODEL_OUTPUT_NAME_PREFIX:-q4binst_q30binst2507_nothink_openscience25k_plain_opd_r4096_b96_n1_lr1e-5}"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
export REWARD_MODEL_INPUT_TOKENIZER="${REWARD_MODEL_INPUT_TOKENIZER:-}"
export STUDENT_CHAT_TEMPLATE_FILE="${STUDENT_CHAT_TEMPLATE_FILE:-}"
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""

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
# The generic runner deliberately drops the incomplete final batch: 25,000 // 96.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-260}"
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
export TEACHER_PREFIX_SFT_LOSS_COEF=0.0
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

# Keep true reward as an observation metric.  ``science_opd_metrics`` returns
# rm_scores for loss and publishes verifier output as true_reward_score.
export DISABLE_CUSTOM_REWARD_FUNCTION=False
export CUSTOM_REWARD_FUNCTION_PATH="scripts/reward/openscience_reasoning2.py"
export CUSTOM_REWARD_FUNCTION_NAME=reward_func
export EXTRA_PPO_ARGS="${EXTRA_PPO_ARGS:-reward_model.reward_manager=science_opd_metrics}"
export SKIP_FINAL_EVAL=True
export TEST_FILE="[\"${PLAIN_TRAIN_DATASET}\"]"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_openscience25k_plain_opd_r4096_b96_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PLAIN_TRAIN_DATASET}"
test -f "${CUSTOM_REWARD_FUNCTION_PATH}"
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"

echo "Science plain token-OPD: ${ACTOR_MODEL_PATH} <- ${REWARD_MODEL_PATH}"
echo "dataset=${PLAIN_TRAIN_DATASET}; steps=${TOTAL_TRAINING_STEPS}; reward_manager=science_opd_metrics"
echo "true reward is metric-only; teacher-derived rm_scores remain the loss reward"

exec bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
