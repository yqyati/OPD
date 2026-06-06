#!/usr/bin/env bash
set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/scripts/data_filter:${PYTHONPATH:-}

INPUT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/datasets/dapo-math-17k-teacher-aligned.parquet
OUTPUT_DIR=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/datasets/opd_prompt_filter
STUDENT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/DeepSeek-R1-Distill-Qwen-1.5B
TEACHER=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/JustRL-DeepSeek-1.5B
BATCH_SIZE=${BATCH_SIZE:-16}

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  --input "${INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --student "${STUDENT}"
  --teacher "${TEACHER}"
  --student-device cuda:0
  --teacher-device cuda:1
  --batch-size "${BATCH_SIZE}"
  --max-length 1024
  --topk 16
  --tail-fraction 0.7
  --min-prefix-tokens 8
  --top-fracs 0.5,0.3
  --dtype bfloat16
  --num-shards 2
)

CUDA_VISIBLE_DEVICES=0,1 python scripts/data_filter/score_prompt_opd_data.py "${COMMON_ARGS[@]}" --shard-index 0 &
PID0=$!

CUDA_VISIBLE_DEVICES=2,3 python scripts/data_filter/score_prompt_opd_data.py "${COMMON_ARGS[@]}" --shard-index 1 &
PID1=$!

wait "${PID0}"
wait "${PID1}"

python scripts/data_filter/merge_prompt_opd_shards.py \
  --input-dir "${OUTPUT_DIR}" \
  --num-shards 2 \
  --topk 16 \
  --top-fracs 0.5,0.3
