#!/bin/bash
set -euo pipefail

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0

python scripts/teacher_prefix/generate_teacher_prefix_data.py \
    --input datasets/opd_prompt_filter/opd_prompt_scores.parquet \
    --output datasets/teacher_prefix/opd_prompt_all_teacher_prefix128.parquet \
    --teacher-model /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/JustRL-DeepSeek-1.5B \
    --gpus 0,1,2,3 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-new-tokens 128 \
    --temperature 0.7 \
    --top-p 0.95 \
    --batch-size 256
