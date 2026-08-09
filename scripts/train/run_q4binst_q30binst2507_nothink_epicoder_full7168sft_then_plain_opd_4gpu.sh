#!/usr/bin/env bash
# EpiCoder 30K pipeline:
# exact 7k 30B teacher responses -> full-response SFT -> plain token OPD.
# The full-response asset is retained so subsequent prefix variants slice the
# exact same token trajectories rather than re-running the teacher.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

SFT_MODEL_NAME="${SFT_MODEL_NAME:-q4binst_q30binst2507_nothink_epicoder30k_full7168_pure_sft_b96_lr1e-5}"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-${SFT_MODEL_NAME}}"
SFT_RUN_TAG="${SFT_RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
SFT_STEP="${SFT_STEP:-312}"

echo "[stage 1/3] Create/reuse exact 7168-token no-think teacher trajectories"
bash scripts/teacher_prefix/run_q4binst_q30binst2507_nothink_epicoder_full7168_4gpu.sh

echo "[stage 2/3] Full7168 pure SFT from the saved teacher trajectories"
FULL_RESPONSE_DATASET="datasets/sft_teacher_response/q4binst_q30binst2507_nothink_epicoder30k_full_response_7168.parquet" \
SFT_DATASET="datasets/sft/q4binst_q30binst2507_nothink_epicoder30k_full7168_pure_sft.parquet" \
SFT_MODEL_NAME="${SFT_MODEL_NAME}" \
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME}" \
SFT_RUN_TAG="${SFT_RUN_TAG}" \
SKIP_FINAL_EVAL=True \
bash scripts/sft/run_q4binst_q30binst2507_nothink_full7168_pure_sft_4gpu.sh

SFT_MODEL_DIR="${OPD_ROOT}/merged_models/${SFT_MODEL_NAME}_step${SFT_STEP}"
test -f "${SFT_MODEL_DIR}/config.json" || {
    echo "SFT did not produce expected merged checkpoint: ${SFT_MODEL_DIR}" >&2
    exit 1
}

echo "[stage 3/3] Plain token OPD from the full7168 SFT checkpoint"
SFT_MODEL_NAME="${SFT_MODEL_NAME}" \
SFT_STEP="${SFT_STEP}" \
SFT_MODEL_DIR="${SFT_MODEL_DIR}" \
exec bash scripts/train/run_q4binst_q30binst2507_nothink_epicoder_full7168sftinit_plain_opd_4gpu.sh
