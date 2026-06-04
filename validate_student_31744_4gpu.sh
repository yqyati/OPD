#!/usr/bin/env bash
set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/scripts/val/data"
OUTPUT_DIR="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/justrl_eval_outputs_31744"
EVAL_DIR="${OUTPUT_DIR}/$(basename "${MODEL_DIR}")"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "Missing model config: ${MODEL_DIR}/config.json" >&2
  exit 1
fi

python scripts/val/eval/gen_vllm.py \
  --model "${MODEL_DIR}" \
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
