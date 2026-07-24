#!/usr/bin/env bash
set -euo pipefail

# One matched study for the actual sequential selector:
# 32 -> 64 -> 128 -> 256 -> 512, with teacher trajectories live at 512.
ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
STUDENT_MODEL="${ROOT}/model/Qwen3-1.7B-Base"
OUTPUT_DIR="${ROOT}/OPD/outputs/handoff_value/q3b_thinking_alive512_tp32_64_128_256_512_k4_p500_seed42_$(date +%Y-%m-%d_%H-%M-%S)"

PREFIX_LENGTHS=32,64,128,256,512
ALIVE_AT_PREFIX_LEN=512
NUM_PROMPTS="${NUM_PROMPTS:-500}"
CONTINUATIONS_PER_PREFIX=4
MAX_TOKENS=4096
TEMPERATURE=1.0
TOP_P=1.0
GPUS="${GPUS:-0,1}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-32}"
SEED=42

test -f "${INPUT_DATASET}"
test -f "${STUDENT_MODEL}/config.json"

echo "========== Completion-Aware Sequential Handoff Study =========="
echo "candidates: ${PREFIX_LENGTHS}; teacher trajectories alive through: ${ALIVE_AT_PREFIX_LEN}"
echo "prompts: ${NUM_PROMPTS}; K per candidate: ${CONTINUATIONS_PER_PREFIX}"
echo "full rollouts: $((NUM_PROMPTS * 5 * CONTINUATIONS_PER_PREFIX)); GPUs: ${GPUS}"
echo "output: ${OUTPUT_DIR}"

python scripts/teacher_prefix/diagnose_handoff_value.py \
  --input "${INPUT_DATASET}" \
  --student-model "${STUDENT_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --prefix-lengths "${PREFIX_LENGTHS}" \
  --require-alive-at-prefix-len "${ALIVE_AT_PREFIX_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --continuations-per-prefix "${CONTINUATIONS_PER_PREFIX}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --gpus "${GPUS}" \
  --request-batch-size "${REQUEST_BATCH_SIZE}" \
  --seed "${SEED}"

python scripts/teacher_prefix/summarize_sequential_handoff_selector.py \
  --rollout-dir "${OUTPUT_DIR}" \
  --prefix-lengths "${PREFIX_LENGTHS}" \
  --required-successes 2 \
  --continuations-per-prefix "${CONTINUATIONS_PER_PREFIX}"
