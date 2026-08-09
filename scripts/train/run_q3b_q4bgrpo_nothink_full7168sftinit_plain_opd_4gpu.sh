#!/usr/bin/env bash
# Plain OPD after the Base-GRPO full7168 pure-SFT warm start.
set -euo pipefail
source .env
cd "${OPD_ROOT}"

SFT_MODEL_NAME="${SFT_MODEL_NAME:-q3b_q4bgrpo_nothink_full7168_pure_sft_b64_lr1e-5}"
SFT_STEP="${SFT_STEP:-279}"
export SFT_MODEL_DIR="${SFT_MODEL_DIR:-${OPD_ROOT}/merged_models/${SFT_MODEL_NAME}_step${SFT_STEP}}"
test -f "${SFT_MODEL_DIR}/config.json" || { echo "Missing merged full7168 SFT model: ${SFT_MODEL_DIR}" >&2; exit 1; }

export RUN_MODE=plain
export PLAIN_TRAIN_DATASET=datasets/dapo-math-17k-teacher-aligned.parquet
export PLAIN_TRAIN_DATASET_NAME=q3b_q4bgrpo_nothink_plain_opd_from_full7168_sft
export PLAIN_MODEL_OUTPUT_NAME_PREFIX=q3b_q4bgrpo_nothink_full7168sftinit_plain_opd_r4096_b64_n1_lr1e-5
export ACTOR_MODEL_PATH="${SFT_MODEL_DIR}"
export REWARD_MODEL_PATH="${MODEL_ROOT}/Qwen3-4B-Base-GRPO"
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
export MINI_BATCH_SIZE=64
export N_RESPONSES=1
export LR="${LR:-1e-5}"
export TOTAL_EPOCHS=1
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
export EVAL_MAX_TOKENS=7168
export EVAL_OUTPUT_DIR="${OPD_ROOT}/outputs/eval/q3b_q4bgrpo_nothink_full7168sftinit_plain"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q3b_q4bgrpo_nothink_full7168sftinit_plain_opd_r4096_b64_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"
exec bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
