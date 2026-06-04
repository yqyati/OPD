#!/usr/bin/env bash
set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0

CKPT_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/checkpoint/token_reward_direct_DAPO-Math-17k-TeacherAligned_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_only_stu-rw_student_p-2026-06-04_07-38-38/global_step_279/actor"
MERGED_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/merged_models/token_reward_direct_DAPO-Math-17k-TeacherAligned_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_only_stu-rw_student_p-2026-06-04_07-38-38_global_step_279"
DATA_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/scripts/val/data"
OUTPUT_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/justrl_eval_outputs"
EVAL_DIR="${OUTPUT_DIR}/$(basename "${MERGED_DIR}")"

# 1. Merge FSDP checkpoint shards into a HuggingFace-format model.
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${CKPT_DIR}" \
  --target_dir "${MERGED_DIR}"

# 2. Generate answers with vLLM.
python scripts/val/eval/gen_vllm.py \
  --model "${MERGED_DIR}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --tasks AIME24,AIME25,AMC23 \
  --n 16 \
  --max-tokens 7168 \
  --temperature 0.7 \
  --top-p 0.95 \
  --gpus 0,1,2,3,4,5,6,7 \
  --disable-thinking

# 3. Grade generated answers.
python scripts/val/eval/grade.py \
  --eval-dir "${EVAL_DIR}"
