#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu
cd "${ROOT}/OPD"

INPUT="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_prefix1024_tokenids_topk16.parquet"
OUTPUT="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_handoff_annotated.parquet"
SHARD_DIR="${ROOT}/OPD/datasets/teacher_prefix/qwen3_4b_base_thinking_dapo_math_17k_teacher_handoff_annotation_shards"
NUM_SHARDS=4
GPU_IDS=${GPU_IDS:-0,1,2,3}
MAX_ROWS=${MAX_ROWS:-0}

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -ne "${NUM_SHARDS}" ]; then
  echo "GPU_IDS must provide exactly ${NUM_SHARDS} comma-separated GPU IDs, got: ${GPU_IDS}" >&2
  exit 2
fi

if [ -d "${SHARD_DIR}" ] && [ -n "$(find "${SHARD_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  if [ "${RESET_SHARDS:-0}" = "1" ]; then
    rm -rf "${SHARD_DIR}"
  else
    echo "Shard directory already contains files: ${SHARD_DIR}" >&2
    echo "After inspecting the failed logs, rerun with RESET_SHARDS=1 to replace them." >&2
    exit 2
  fi
fi
mkdir -p "${SHARD_DIR}"

pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="${GPUS[${shard_id}]}" VLLM_USE_FLASHINFER_SAMPLER=0 \
  python scripts/teacher_prefix/annotate_teacher_handoff_boundaries.py \
    --input "${INPUT}" \
    --output "${SHARD_DIR}/handoff_annotation_shard_$(printf '%02d' "${shard_id}")_of_$(printf '%02d' "${NUM_SHARDS}").parquet" \
    --teacher-model "${ROOT}/model/Qwen3-4B-Base" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --batch-size 96 \
    --max-retries 2 \
    --max-rows "${MAX_ROWS}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard_id}" \
    > "${SHARD_DIR}/shard_${shard_id}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  if ! wait "${pids[${shard_id}]}"; then
    echo "Annotation shard ${shard_id} failed; see ${SHARD_DIR}/shard_${shard_id}.log" >&2
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  exit 1
fi

python scripts/teacher_prefix/merge_teacher_handoff_annotation_shards.py \
  --input-dir "${SHARD_DIR}" \
  --num-shards "${NUM_SHARDS}" \
  --output "${OUTPUT}"
