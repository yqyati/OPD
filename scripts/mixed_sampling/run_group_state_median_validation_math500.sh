#!/usr/bin/env bash
source .env

set -euo pipefail

# External validation of the Group-OPD state ranking metric on MATH-500.
# MATH-500 is never used by the DAPO-Math-17k training runs.
ROOT=${YANGQINGYU_ROOT}
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${ROOT}/OPD/datasets/test_data/MATH-500/test.parquet"
STUDENT_MODEL="${ROOT}/model/Qwen3-1.7B-Base"
TEACHER_MODEL="${ROOT}/model/Qwen3-4B-Base"
RUN_ID="q3b_q4b_math500_group_prefix128_n4_k4_seed42_$(date +%Y-%m-%d_%H-%M-%S)"
STATE_DIR="${ROOT}/OPD/outputs/group_state_value/${RUN_ID}"
METRIC_DIR="${ROOT}/OPD/outputs/group_state_metric_benchmark/${RUN_ID}"

NUM_PROMPTS=500
PREFIXES_PER_PROMPT=4
PREFIX_TOKENS=128
CONTINUATIONS_PER_PREFIX=4
MAX_TOKENS=4096
TEMPERATURE=1.0
TOP_P=1.0
GPUS=0,1,2,3
ROLLOUT_BATCH_SIZE=32
FORWARD_BATCH_SIZE=4
SEED=42

test -f "${INPUT_DATASET}"
test -f "${STUDENT_MODEL}/config.json"
test -f "${TEACHER_MODEL}/config.json"

echo "========== External Group-State Validation: MATH-500 =========="
echo "states: ${NUM_PROMPTS} prompts x ${PREFIXES_PER_PROMPT} prefixes; K=${CONTINUATIONS_PER_PREFIX}"
echo "full continuations: $((NUM_PROMPTS * PREFIXES_PER_PROMPT * CONTINUATIONS_PER_PREFIX))"
echo "state output: ${STATE_DIR}"

python scripts/mixed_sampling/diagnose_group_student_state_value.py \
  --input "${INPUT_DATASET}" \
  --student-model "${STUDENT_MODEL}" \
  --output-dir "${STATE_DIR}" \
  --num-prompts "${NUM_PROMPTS}" \
  --prefixes-per-prompt "${PREFIXES_PER_PROMPT}" \
  --prefix-tokens "${PREFIX_TOKENS}" \
  --continuations-per-prefix "${CONTINUATIONS_PER_PREFIX}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --gpus "${GPUS}" \
  --request-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --seed "${SEED}"

echo "========== Forward-only Metric Benchmark =========="
python scripts/mixed_sampling/benchmark_group_state_ranking_metrics.py \
  --input "${INPUT_DATASET}" \
  --state-values "${STATE_DIR}/group_prefix_state_values.jsonl" \
  --continuations "${STATE_DIR}/group_prefix_continuations.jsonl" \
  --student-model "${STUDENT_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${METRIC_DIR}" \
  --gpus "${GPUS}" \
  --batch-size "${FORWARD_BATCH_SIZE}"
