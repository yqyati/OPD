#!/usr/bin/env bash
# Two-stage Instruct control: evaluate full-7168 pure SFT, then evaluate plain
# OPD initialized from that exact SFT checkpoint.  Never submits an rjob.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

SFT_MODEL_NAME="${SFT_MODEL_NAME:-q4binst_q30binst2507_nothink_full7168_pure_sft_b96_lr1e-5}"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-${SFT_MODEL_NAME}}"
SFT_RUN_TAG="${SFT_RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"

echo "[stage 1/2] Full-7168 teacher-response pure SFT, merge, and 9192-token eval"
SFT_MODEL_NAME="${SFT_MODEL_NAME}" \
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME}" \
SFT_RUN_TAG="${SFT_RUN_TAG}" \
bash scripts/sft/run_q4binst_q30binst2507_nothink_full7168_pure_sft_4gpu.sh

# 17,917 DAPO rows / batch 96 / one epoch = floor(17917 / 96) = 186 steps.
SFT_STEP="${SFT_STEP:-186}"
SFT_MODEL_DIR="${OPD_ROOT}/merged_models/${SFT_MODEL_NAME}_step${SFT_STEP}"
test -f "${SFT_MODEL_DIR}/config.json" || {
    echo "Stage 1 did not produce expected merged SFT model: ${SFT_MODEL_DIR}" >&2
    exit 1
}

echo "[stage 2/2] Plain OPD from evaluated full-SFT checkpoint, then 9192-token eval"
SFT_MODEL_NAME="${SFT_MODEL_NAME}" \
SFT_STEP="${SFT_STEP}" \
SFT_MODEL_DIR="${SFT_MODEL_DIR}" \
exec bash scripts/train/run_q4binst_q30binst2507_nothink_full7168sftinit_plain_opd_4gpu.sh
