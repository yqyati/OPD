#!/usr/bin/env bash
source .env

set -euo pipefail
set -x

cd ${OPD_ROOT}

export PYTHONPATH=${OPD_ROOT}/verl:${OPD_ROOT}/scripts/data_filter:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

INPUT=${OPD_ROOT}/datasets/dapo-math-17k-teacher-aligned.parquet
PROMPT_SCORES=${OPD_ROOT}/datasets/opd_prompt_filter/opd_prompt_scores.parquet
OUTPUT_DIR=${OPD_ROOT}/datasets/opd_prefix50_filter
PREFIX_ROLLOUTS=${OUTPUT_DIR}/student_prefix50_rollouts.parquet
STUDENT=${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B
TEACHER=${MODEL_ROOT}/JustRL-DeepSeek-1.5B
BATCH_SIZE=${BATCH_SIZE:-8}
PYTHON=${PYTHON:-/root/miniconda3/envs/verl/bin/python}
GEN_GPUS=${GEN_GPUS:-0}
GEN_TP=${GEN_TP:-1}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-256}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" scripts/data_filter/generate_prefix50_rollouts.py \
  --input "${INPUT}" \
  --output "${PREFIX_ROLLOUTS}" \
  --model "${STUDENT}" \
  --gpus "${GEN_GPUS}" \
  --tensor-parallel-size "${GEN_TP}" \
  --max-model-len 2048 \
  --max-new-tokens 50 \
  --temperature 1.0 \
  --top-p 0.95 \
  --batch-size "${GEN_BATCH_SIZE}"

CUDA_VISIBLE_DEVICES=0,1 "${PYTHON}" scripts/data_filter/score_prompt_prefix50_opd_data.py \
  --input "${INPUT}" \
  --prompt-scores "${PROMPT_SCORES}" \
  --prefix-rollouts "${PREFIX_ROLLOUTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --student "${STUDENT}" \
  --teacher "${TEACHER}" \
  --student-device cuda:0 \
  --teacher-device cuda:1 \
  --batch-size "${BATCH_SIZE}" \
  --max-length 1536 \
  --topk 16 \
  --top-fracs 0.5,0.3 \
  --dtype bfloat16
