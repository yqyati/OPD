#!/usr/bin/env bash
# Same-origin Prefix-128 MOPD with prefix-SFT coefficient 0.2.
set -euo pipefail
SCRIPT_DIR=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/OPD/scripts/mopd
RUN_NAME=q8b_sameorigin_prefix128_sft02_mopd_r7168_n1_b140_mb40_ep1_lr1e-6 \
PREFIX_SFT_COEF=0.2 \
  exec bash "${SCRIPT_DIR}/train_sameorigin_prefix128_mopd_r7168_8gpu.bash"
