#!/bin/bash
source .env

set -euo pipefail

cd ${OPD_ROOT}

export PYTHONPATH=${OPD_ROOT}/verl:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0

python scripts/teacher_prefix/generate_teacher_prefix_data.py \
    --input datasets/dapo-math-17k-teacher-aligned.parquet \
    --output datasets/teacher_prefix/qwen3_grpo_dapo_math_17k_teacher_prefix128.parquet \
    --teacher-model ${MODEL_ROOT}/Qwen3-4B-Base-GRPO \
    --gpus 0,1,2,3,4,5,6,7 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-new-tokens 128 \
    --temperature 0.7 \
    --top-p 0.95 \
    --batch-size 256
