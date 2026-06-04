#!/usr/bin/env bash
set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0

RUN_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/checkpoint/token_reward_direct_DAPO-Math-17k-TeacherAligned_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_only_stu-rw_student_p-2026-06-04_07-38-38"
CKPT_DIR="${RUN_DIR}/global_step_140/actor"
MERGED_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/merged_models/opd_teacheraligned_step140"
DATA_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/scripts/val/data"
OUTPUT_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/justrl_eval_outputs_31744"
EVAL_DIR="${OUTPUT_DIR}/$(basename "${MERGED_DIR}")"

if ! ls "${CKPT_DIR}"/model_world_size_*_rank_0.pt >/dev/null 2>&1; then
  echo "Missing FSDP checkpoint shard: ${CKPT_DIR}/model_world_size_*_rank_0.pt" >&2
  exit 1
fi

if [ ! -f "${MERGED_DIR}/config.json" ]; then
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${MERGED_DIR}"
fi

python scripts/val/eval/gen_vllm.py \
  --model "${MERGED_DIR}" \
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
