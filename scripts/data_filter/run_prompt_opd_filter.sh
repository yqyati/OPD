#!/usr/bin/env bash
source .env

set -euo pipefail
set -x

cd ${OPD_ROOT}

export PYTHONPATH=${OPD_ROOT}/verl:${PYTHONPATH:-}

python scripts/data_filter/score_prompt_opd_data.py \
  --input ${OPD_ROOT}/datasets/dapo-math-17k-teacher-aligned.parquet \
  --output-dir ${OPD_ROOT}/datasets/opd_prompt_filter \
  --student ${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B \
  --teacher ${MODEL_ROOT}/JustRL-DeepSeek-1.5B \
  --student-device cuda:0 \
  --teacher-device cuda:1 \
  --batch-size 8 \
  --max-length 1024 \
  --topk 16 \
  --tail-fraction 0.7 \
  --min-prefix-tokens 8 \
  --top-fracs 0.5,0.3 \
  --dtype bfloat16
