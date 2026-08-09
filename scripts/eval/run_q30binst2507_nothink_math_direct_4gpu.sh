#!/usr/bin/env bash
# Direct math evaluation only (no training / no teacher rollout):
# Qwen3-30B-A3B-Instruct-2507 with its native Qwen3 Instruct template,
# no-thinking mode.  One TP=1 vLLM worker is placed on each of four GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
source .env
cd "${OPD_ROOT}"

export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
# This environment has no nvcc for FlashInfer JIT.  Keep the sampler used by
# all existing math evaluations.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL_DIR="${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_DIR="${OPD_ROOT}/outputs/eval/qwen3_30b_a3b_instruct2507_nothink_direct_9192"
EVAL_DIR="${OUTPUT_DIR}/$(basename "${MODEL_DIR}")"

test -f "${MODEL_DIR}/config.json" || {
    echo "Missing teacher model config: ${MODEL_DIR}/config.json" >&2
    exit 1
}

echo "Math teacher direct evaluation: Qwen3-30B-A3B-Instruct-2507"
echo "Protocol: native template, no-thinking; AMC23/AIME24/AIME25; n=16, T=0.7, top-p=0.95"
echo "Four data-parallel workers: GPUs 0,1,2,3; TP=1 each"
echo "Output directory: ${EVAL_DIR}"

python scripts/val/eval/gen_vllm.py \
    --model "${MODEL_DIR}" \
    --data-dir "${OPD_ROOT}/scripts/val/data" \
    --output-dir "${OUTPUT_DIR}" \
    --tasks AIME24,AIME25,AMC23 \
    --n 16 \
    --max-tokens 9192 \
    --temperature 0.7 \
    --top-p 0.95 \
    --gpus 0,1,2,3 \
    --disable-thinking

python scripts/val/eval/grade.py \
    --eval-dir "${EVAL_DIR}"
