#!/usr/bin/env bash
set -euo pipefail

STEP=${STEP:-139}
RUN_NAME="token_reward_direct_DAPO-Math-17k-TeacherAligned-Top50_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_7168-T_1.0-Tch_1.0-n_4-mbs_64-lr_1e-5-topk_16-topk_strategy_only_stu-rw_student_p-2026-06-06_08-42-07"

CKPT_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/checkpoint/${RUN_NAME}/global_step_${STEP}/actor"
MERGED_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/merged_models/opd_top50_lr1e-5_step${STEP}"

/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/scripts/validate/eval_checkpoint.sh \
  --ckpt-dir "${CKPT_DIR}" \
  --merged-dir "${MERGED_DIR}" \
  --output-dir /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/outputs/eval/justrl_eval_outputs_31744 \
  --max-tokens 31744 \
  --gpus 0,1,2,3
