#!/usr/bin/env bash
# EpiCoder plain OPD initialized from the EpiCoder full-7168 pure-SFT model.
# No code verifier is launched; only teacher-derived token rm_scores are used.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

SFT_MODEL_NAME="${SFT_MODEL_NAME:-q4binst_q30binst2507_nothink_epicoder30k_full7168_pure_sft_b96_lr1e-5}"
SFT_STEP="${SFT_STEP:-312}"
export SFT_MODEL_DIR="${SFT_MODEL_DIR:-${OPD_ROOT}/merged_models/${SFT_MODEL_NAME}_step${SFT_STEP}}"
test -f "${SFT_MODEL_DIR}/config.json" || { echo "Missing merged EpiCoder SFT model: ${SFT_MODEL_DIR}" >&2; exit 1; }

export RUN_MODE=plain
export PLAIN_TRAIN_DATASET="datasets/epicoder-func-380k/epicoder_func_30k_seed42_verl.parquet"
export PLAIN_TRAIN_DATASET_NAME="EpiCoder-func-380k-30K-seed42-4BInst-30BInst-NoThink-Full7168SFTInit-PlainOPD"
export PLAIN_MODEL_OUTPUT_NAME_PREFIX="q4binst_q30binst2507_nothink_epicoder30k_full7168sftinit_plain_opd_r4096_b96_n1_lr1e-5"
export ACTOR_MODEL_PATH="${SFT_MODEL_DIR}"
export REWARD_MODEL_PATH="${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507"
export REWARD_MODEL_INPUT_TOKENIZER=""
export STUDENT_CHAT_TEMPLATE_FILE=""
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""
export TEACHER_PREFIX_MAX_LEN=0
export TEACHER_PREFIX_SFT_LOSS_COEF=0.0
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
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-312}"
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
export DISABLE_CUSTOM_REWARD_FUNCTION=True
export EXTRA_PPO_ARGS="reward_model.reward_manager=batch"
export SKIP_FINAL_EVAL=True
export TEST_FILE="[\"${PLAIN_TRAIN_DATASET}\"]"
export RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_epicoder30k_full7168sftinit_plain_opd_r4096_b96_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

echo "[plain OPD] EpiCoder from full7168 SFT checkpoint; no CPU code verification"
exec bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
