#!/usr/bin/env bash
set -euo pipefail

# Completion-aware teacher-prefix handoff comparison. All selected teacher
# trajectories remain live through token 512, so 128/256/512 are matched
# suffix handoff states rather than EOS-contaminated continuations.
ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"

export PYTHONPATH="${ROOT}/OPD/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
STUDENT_MODEL="${ROOT}/model/Qwen3-1.7B-Base"
OUTPUT_DIR="${ROOT}/OPD/outputs/handoff_value/q3b_thinking_alive512_tp128_256_512_k8_p500_seed42_$(date +%Y-%m-%d_%H-%M-%S)"

PREFIX_LENGTHS=128,256,512
ALIVE_AT_PREFIX_LEN=512
NUM_PROMPTS=500
CONTINUATIONS_PER_PREFIX=8
MAX_TOKENS=4096
TEMPERATURE=1.0
TOP_P=1.0
GPUS=0,1,2,3
REQUEST_BATCH_SIZE=32
SEED=42

test -f "${INPUT_DATASET}"
test -f "${STUDENT_MODEL}/config.json"

echo "========== Completion-Aware Handoff Value: Prefix128 vs Prefix256 vs Prefix512 =========="
echo "teacher trajectories: alive at token ${ALIVE_AT_PREFIX_LEN}; matched prompt subset only"
echo "prompts: ${NUM_PROMPTS}; K: ${CONTINUATIONS_PER_PREFIX}; full rollouts: $((NUM_PROMPTS * 3 * CONTINUATIONS_PER_PREFIX))"
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
