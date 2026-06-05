#!/usr/bin/env bash
set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0

STEP=${STEP:-260}
RUN_NAME="token_reward_direct_DAPO-Math-17k-TeacherAligned_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_7168-T_1.0-Tch_1.0-n_4-mbs_64-lr_1e-5-topk_16-topk_strategy_only_stu-rw_student_p-2026-06-05_14-55-47"

CKPT_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/checkpoint/${RUN_NAME}/global_step_${STEP}/actor"
MODEL_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/merged_models/opd_teacheraligned_lr1e-5_step${STEP}"
DATA_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/scripts/val/data"
OUTPUT_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/justrl_eval_outputs_31744"
EVAL_DIR="${OUTPUT_DIR}/$(basename "${MODEL_DIR}")"

if [ ! -d "${CKPT_DIR}" ]; then
  echo "Missing checkpoint dir: ${CKPT_DIR}" >&2
  exit 1
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${MODEL_DIR}"
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "Missing model config: ${MODEL_DIR}/config.json" >&2
  exit 1
fi

python scripts/val/eval/gen_vllm.py \
  --model "${MODEL_DIR}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --tasks AIME24,AIME25,AMC23 \
  --n 16 \
  --max-tokens 31744 \
  --temperature 0.7 \
  --top-p 0.95 \
  --gpus 0,1,2,3 \
  --disable-thinking

python scripts/val/eval/grade.py \
  --eval-dir "${EVAL_DIR}"
