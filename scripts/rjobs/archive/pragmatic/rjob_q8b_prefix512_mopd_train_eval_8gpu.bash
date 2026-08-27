#!/usr/bin/env bash
# Train Prefix-512 MOPD, merge it, then evaluate Math, Code, and IFEval.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl
source /mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/OPD/.env

YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
TRAIN_LAUNCHER="${YANGQINGYU_ROOT}/workspace/OPD/scripts/rjobs/archive/pragmatic/rjob_launch_q8b_pragmatic_prefix512_mopd_8gpu.bash"
EVAL_LAUNCHER="${YANGQINGYU_ROOT}/rjob_eval_q8b_mopd_plain_then_prefix128_8gpu.bash"
MERGED_MODEL="${YANGQINGYU_ROOT}/workspace/OPD/merged_models/q8b_pragmatic_prefix512_mopd_8gpu_step484"

test -f "${TRAIN_LAUNCHER}" || { echo "Missing training launcher: ${TRAIN_LAUNCHER}" >&2; exit 1; }
test -f "${EVAL_LAUNCHER}" || { echo "Missing evaluation launcher: ${EVAL_LAUNCHER}" >&2; exit 1; }

echo "[prefix512 1/2] Train, checkpoint, and merge Prefix-512 MOPD"
bash "${TRAIN_LAUNCHER}"
test -f "${MERGED_MODEL}/config.json" || { echo "Prefix-512 MOPD merge is incomplete: ${MERGED_MODEL}" >&2; exit 1; }
compgen -G "${MERGED_MODEL}/*.safetensors" >/dev/null || { echo "Prefix-512 MOPD weights are missing: ${MERGED_MODEL}" >&2; exit 1; }

echo "[prefix512 2/2] Evaluate Math, Code, and Instruction on all 8 GPUs"
EVAL_TARGET=prefix512 PREFIX512_MODEL="merged_models/q8b_pragmatic_prefix512_mopd_8gpu_step484" bash "${EVAL_LAUNCHER}"

echo "Prefix-512 MOPD training and evaluation completed."
