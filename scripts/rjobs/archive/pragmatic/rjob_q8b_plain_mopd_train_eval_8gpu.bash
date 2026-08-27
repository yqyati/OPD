#!/usr/bin/env bash
# Train plain MOPD, publish its final model, then evaluate Math/Code/IFEval.
set -euo pipefail

YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
TRAIN_LAUNCHER="${YANGQINGYU_ROOT}/workspace/OPD/scripts/rjobs/archive/pragmatic/rjob_launch_q8b_pragmatic_plain_mopd_8gpu.bash"
EVAL_LAUNCHER="${YANGQINGYU_ROOT}/rjob_eval_q8b_mopd_plain_then_prefix128_8gpu.bash"
MERGED_MODEL="${YANGQINGYU_ROOT}/workspace/OPD/merged_models/q8b_pragmatic_plain_mopd_8gpu_step484"

test -f "${TRAIN_LAUNCHER}" || { echo "Missing training launcher: ${TRAIN_LAUNCHER}" >&2; exit 1; }
test -f "${EVAL_LAUNCHER}" || { echo "Missing evaluation launcher: ${EVAL_LAUNCHER}" >&2; exit 1; }

echo "[plain 1/2] Train, checkpoint, and merge plain MOPD"
bash "${TRAIN_LAUNCHER}"
test -f "${MERGED_MODEL}/config.json" || { echo "Plain MOPD merge is incomplete: ${MERGED_MODEL}" >&2; exit 1; }
compgen -G "${MERGED_MODEL}/*.safetensors" >/dev/null || { echo "Plain MOPD weights are missing: ${MERGED_MODEL}" >&2; exit 1; }

echo "[plain 2/2] Evaluate Math, Code, and Instruction on all 8 GPUs"
EVAL_TARGET=plain bash "${EVAL_LAUNCHER}"

echo "Plain MOPD training and evaluation completed."
