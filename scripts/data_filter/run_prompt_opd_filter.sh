#!/usr/bin/env bash
set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:${PYTHONPATH:-}

python scripts/data_filter/score_prompt_opd_data.py \
  --input /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/datasets/dapo-math-17k-teacher-aligned.parquet \
  --output-dir /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/datasets/opd_prompt_filter \
  --student /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/DeepSeek-R1-Distill-Qwen-1.5B \
  --teacher /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/JustRL-DeepSeek-1.5B \
  --student-device cuda:0 \
  --teacher-device cuda:1 \
  --batch-size 8 \
  --max-length 1024 \
  --topk 16 \
  --tail-fraction 0.7 \
  --min-prefix-tokens 8 \
  --top-fracs 0.5,0.3 \
  --dtype bfloat16
