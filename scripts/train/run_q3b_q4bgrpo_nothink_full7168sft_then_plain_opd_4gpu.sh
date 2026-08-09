#!/usr/bin/env bash
# Base-GRPO full7168 pure SFT -> plain OPD sequence.
set -euo pipefail
source .env
cd "${OPD_ROOT}"

echo "[stage 1/3] Generate/reuse exact 7168-token-cap teacher trajectories"
bash scripts/teacher_prefix/run_q3b_q4bgrpo_nothink_full7168_4gpu.sh

SFT_RUN_TAG="${SFT_RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
SFT_MODEL_NAME="${SFT_MODEL_NAME:-q3b_q4bgrpo_nothink_full7168_pure_sft_b64_lr1e-5}"
echo "[stage 2/3] Full7168 pure SFT, merge, and eval"
FULL_RESPONSE_MAX_TOKENS=7168 \
FULL_RESPONSE_DATASET=datasets/sft_teacher_response/q3b_q4bgrpo_nothink_dapo_math17k_full_response_7168.parquet \
SFT_EXPERIMENT_NAME="${SFT_MODEL_NAME}" \
SFT_MODEL_NAME="${SFT_MODEL_NAME}" \
SFT_RUN_TAG="${SFT_RUN_TAG}" \
bash scripts/sft/run_q3b_q4bgrpo_nothink_full4096_pure_sft_4gpu.sh

SFT_STEP="${SFT_STEP:-279}"
SFT_MODEL_DIR="${OPD_ROOT}/merged_models/${SFT_MODEL_NAME}_step${SFT_STEP}"
test -f "${SFT_MODEL_DIR}/config.json" || { echo "Missing merged SFT model: ${SFT_MODEL_DIR}" >&2; exit 1; }

echo "[stage 3/3] Plain OPD from full7168 SFT checkpoint, merge, and eval"
SFT_MODEL_NAME="${SFT_MODEL_NAME}" SFT_STEP="${SFT_STEP}" SFT_MODEL_DIR="${SFT_MODEL_DIR}" \
exec bash scripts/train/run_q3b_q4bgrpo_nothink_full7168sftinit_plain_opd_4gpu.sh
