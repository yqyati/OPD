#!/usr/bin/env bash
# Two-stage control: evaluate full-4096 pure SFT, then evaluate plain OPD
# initialized from that exact SFT checkpoint.  Run only inside an allocated
# four-GPU rjob; this script never submits an rjob itself.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

SFT_MODEL_NAME="${SFT_MODEL_NAME:-q3b_q4bgrpo_nothink_full4096_pure_sft_b64_lr1e-5}"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-${SFT_MODEL_NAME}}"
SFT_RUN_TAG="${SFT_RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"

echo "[stage 1/2] Full teacher-response pure SFT (teacher completion cap=4096), merge, and eval"
SFT_MODEL_NAME="${SFT_MODEL_NAME}" \
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME}" \
SFT_RUN_TAG="${SFT_RUN_TAG}" \
bash scripts/sft/run_q3b_q4bgrpo_nothink_full4096_pure_sft_4gpu.sh

# 17,917 DAPO rows / batch 64 / one epoch = floor(17917 / 64) = 279 steps.
SFT_STEP="${SFT_STEP:-279}"
SFT_MODEL_DIR="${OPD_ROOT}/merged_models/${SFT_MODEL_NAME}_step${SFT_STEP}"
test -f "${SFT_MODEL_DIR}/config.json" || {
    echo "Stage 1 did not produce expected merged SFT model: ${SFT_MODEL_DIR}" >&2
    exit 1
}

echo "[stage 2/2] Plain OPD from the evaluated full-SFT checkpoint, then merge and eval"
SFT_MODEL_NAME="${SFT_MODEL_NAME}" \
SFT_STEP="${SFT_STEP}" \
SFT_MODEL_DIR="${SFT_MODEL_DIR}" \
exec bash scripts/train/run_q3b_q4bgrpo_nothink_fullsftinit_plain_opd_4gpu.sh
