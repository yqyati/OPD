#!/usr/bin/env bash
source .env

set -euo pipefail
set -x

cd ${OPD_ROOT}

export PYTHONPATH=${OPD_ROOT}/verl:${OPD_ROOT}/scripts/data_filter:${PYTHONPATH:-}

INPUT=${OPD_ROOT}/datasets/dapo-math-17k-teacher-aligned.parquet
OUTPUT_DIR=${OPD_ROOT}/datasets/opd_prompt_filter
STUDENT=${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B
TEACHER=${MODEL_ROOT}/JustRL-DeepSeek-1.5B
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
