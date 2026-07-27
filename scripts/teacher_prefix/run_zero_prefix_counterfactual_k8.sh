#!/usr/bin/env bash
source .env

set -euo pipefail

# Generate no-teacher-prefix K=8 rollouts for exactly the same prompts as the
# completed five-length teacher-prefix K=8 study.
ROOT=${YANGQINGYU_ROOT}
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
STUDENT_MODEL="${ROOT}/model/Qwen3-1.7B-Base"
PREFIX_STUDY_DIR="${ROOT}/OPD/outputs/handoff_value/q3b_thinking_alive512_tp32_64_128_256_512_k8_p500_seed42_2026-07-17_22-03-46"
OUTPUT_DIR="${ROOT}/OPD/outputs/handoff_value/q3b_thinking_alive512_zero_prefix_k8_p500_seed42_$(date +%Y-%m-%d_%H-%M-%S)"

NUM_PROMPTS=500
CONTINUATIONS_PER_PREFIX=8
MAX_TOKENS=4096
TEMPERATURE=1.0
TOP_P=1.0
GPUS="${GPUS:-0,1}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-48}"
SEED=42

test -f "${INPUT_DATASET}"
test -f "${STUDENT_MODEL}/config.json"
test -f "${PREFIX_STUDY_DIR}/config.json"

echo "========== No-Prefix Counterfactual Rollouts (K=8) =========="
echo "reusing exact source rows from: ${PREFIX_STUDY_DIR}"
echo "prompts: ${NUM_PROMPTS}; K: ${CONTINUATIONS_PER_PREFIX}; full rollouts: $((NUM_PROMPTS * CONTINUATIONS_PER_PREFIX))"
echo "output: ${OUTPUT_DIR}"

python scripts/teacher_prefix/diagnose_handoff_value.py \
  --input "${INPUT_DATASET}" \
  --student-model "${STUDENT_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --prefix-lengths 0 \
  --require-alive-at-prefix-len 512 \
  --selected-source-rows-from "${PREFIX_STUDY_DIR}/config.json" \
  --num-prompts "${NUM_PROMPTS}" \
  --continuations-per-prefix "${CONTINUATIONS_PER_PREFIX}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --gpus "${GPUS}" \
  --request-batch-size "${REQUEST_BATCH_SIZE}" \
  --seed "${SEED}"
