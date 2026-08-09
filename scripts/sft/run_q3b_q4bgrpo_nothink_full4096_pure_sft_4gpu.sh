#!/usr/bin/env bash
# Pure full-response SFT control for Qwen3-1.7B-Base <- Qwen3-4B-Base-GRPO.
# The exact saved no-think teacher token IDs are supervised up to their natural
# EOS or the original 4096-token rollout cap; no teacher prefix is supplied at
# inference or in a subsequent OPD stage.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

FULL_RESPONSE_MAX_TOKENS="${FULL_RESPONSE_MAX_TOKENS:-4096}"
FULL_RESPONSE_DATASET="${FULL_RESPONSE_DATASET:-datasets/sft_teacher_response/q3b_q4bgrpo_nothink_dapo_math17k_full_response_${FULL_RESPONSE_MAX_TOKENS}.parquet}"
SFT_DATASET="${SFT_DATASET:-datasets/sft/q3b_q4bgrpo_nothink_full${FULL_RESPONSE_MAX_TOKENS}_pure_sft.parquet}"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-q3b_q4bgrpo_nothink_full${FULL_RESPONSE_MAX_TOKENS}_pure_sft_b64_lr1e-5}"
SFT_MODEL_NAME="${SFT_MODEL_NAME:-q3b_q4bgrpo_nothink_full${FULL_RESPONSE_MAX_TOKENS}_pure_sft_b64_lr1e-5}"
SFT_RUN_TAG="${SFT_RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"

test -f "${FULL_RESPONSE_DATASET}" || {
    echo "Missing ${FULL_RESPONSE_MAX_TOKENS}-token teacher full-response data: ${FULL_RESPONSE_DATASET}" >&2
    exit 1
}

# `MAX_LENGTH` is the complete SFT sequence length: at most 2048 prompt
# tokens plus the teacher's at-most-4096 generated tokens.  The target itself
# remains capped at 4096 by the saved full-response asset.
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-1.7B-Base}" \
SOURCE_PREFIX_DATA="${FULL_RESPONSE_DATASET}" \
SFT_DATASET="${SFT_DATASET}" \
RESPONSE_COLUMN=teacher_response_text \
USE_GENERATED_TOKEN_IDS=True \
GENERATED_TOKEN_IDS_COLUMN=teacher_response_token_ids \
GENERATED_FINISH_REASON_COLUMN=teacher_response_finish_reason \
ENABLE_THINKING=False \
STUDENT_CHAT_TEMPLATE_FILE="" \
SOURCE_EOS_TOKEN_ID="" \
CANONICAL_EOS_TOKEN_ID="" \
MAX_LENGTH="$((2048 + FULL_RESPONSE_MAX_TOKENS))" \
TRAIN_BATCH_SIZE=64 \
MICRO_BATCH_SIZE_PER_GPU=1 \
N_GPUS_PER_NODE=4 \
EVAL_GPUS=0,1,2,3 \
LR="${LR:-1e-5}" \
TOTAL_EPOCHS=1 \
DATA_SEED=42 \
EVAL_MAX_TOKENS=7168 \
EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME}" \
CKPT_PATH="checkpoint/${SFT_EXPERIMENT_NAME}_${SFT_RUN_TAG}" \
MODEL_NAME="${SFT_MODEL_NAME}" \
OUTPUT_DIR="${OPD_ROOT}/outputs/eval/q3b_q4bgrpo_nothink_full4096_pure_sft" \
bash scripts/sft/run_qwen3_grpo_teacher_prefix128_pure_sft_4gpu.sh
