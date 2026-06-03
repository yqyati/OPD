#!/usr/bin/env bash
set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/verl:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0

CKPT_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/checkpoint/official_opd_DAPO-Math-17k_Qwen3-1.7B-SFT_from_Qwen3-4B-Base-GRPO_4096_n1_mbs112_topk64_2026-06-03_15-02-27/global_step_318/actor"
MERGED_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/merged_models/official_opd_global_step_318_4096"
DATA_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/scripts/val/data"
OUTPUT_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/justrl_eval_outputs"
EVAL_DIR="${OUTPUT_DIR}/official_opd_global_step_318_4096"

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
  --max-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.95 \
  --gpus 0,1,2,3,4,5,6,7 \
  --disable-thinking

# 3. Grade generated answers.
python scripts/val/eval/grade.py \
  --eval-dir "${EVAL_DIR}"
