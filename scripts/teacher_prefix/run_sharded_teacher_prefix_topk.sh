#!/usr/bin/env bash
set -euo pipefail

: "${TOPK_INPUT:?Set TOPK_INPUT}"
: "${TOPK_OUTPUT:?Set TOPK_OUTPUT}"
: "${TOPK_TEACHER_MODEL:?Set TOPK_TEACHER_MODEL}"
: "${TOPK_GPU_GROUPS:?Set TOPK_GPU_GROUPS, e.g. '0;1;2;3'}"

TOPK_K=${TOPK_K:-16}
TOPK_BATCH_SIZE=${TOPK_BATCH_SIZE:-16}
TOPK_MAX_LENGTH=${TOPK_MAX_LENGTH:-2048}
TOPK_ENABLE_THINKING=${TOPK_ENABLE_THINKING:-True}

case "${TOPK_ENABLE_THINKING,,}" in
    true) THINKING_ARG=(--enable-thinking) ;;
    false) THINKING_ARG=() ;;
    *) echo "TOPK_ENABLE_THINKING must be True or False" >&2; exit 1 ;;
esac

if [ -f "${TOPK_OUTPUT}" ]; then
    echo "Teacher-prefix top-k dataset already exists; validating: ${TOPK_OUTPUT}"
    python scripts/teacher_prefix/validate_teacher_prefix_topk.py \
        --input "${TOPK_INPUT}" \
        --output "${TOPK_OUTPUT}" \
        --top-k "${TOPK_K}"
    exit 0
fi

IFS=';' read -r -a GPU_GROUPS <<< "${TOPK_GPU_GROUPS}"
NUM_SHARDS=${#GPU_GROUPS[@]}
SHARD_DIR="${TOPK_OUTPUT%.parquet}_shards"
mkdir -p "${SHARD_DIR}"

TOPK_INPUT="${TOPK_INPUT}" SHARD_DIR="${SHARD_DIR}" NUM_SHARDS="${NUM_SHARDS}" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

df = pd.read_parquet(os.environ["TOPK_INPUT"]).reset_index(drop=True)
df["__opd_original_index"] = range(len(df))
out = Path(os.environ["SHARD_DIR"])
n = int(os.environ["NUM_SHARDS"])
for rank in range(n):
    start = len(df) * rank // n
    end = len(df) * (rank + 1) // n
    path = out / f"input_shard_{rank:02d}_of_{n:02d}.parquet"
    if not path.exists():
        df.iloc[start:end].to_parquet(path, index=False)
    print(f"input shard {rank:02d}: rows={end-start} range=[{start},{end})")
PY

pids=()
outputs=()
for rank in "${!GPU_GROUPS[@]}"; do
    group=${GPU_GROUPS[$rank]}
    input=$(printf '%s/input_shard_%02d_of_%02d.parquet' "${SHARD_DIR}" "$rank" "${NUM_SHARDS}")
    output=$(printf '%s/topk_shard_%02d_of_%02d.parquet' "${SHARD_DIR}" "$rank" "${NUM_SHARDS}")
    log=$(printf '%s/topk_shard_%02d_of_%02d.log' "${SHARD_DIR}" "$rank" "${NUM_SHARDS}")
    outputs+=("${output}")
    if [ -f "${output}" ]; then
        echo "reuse completed top-k shard ${rank}/${NUM_SHARDS}: ${output}"
        pids+=("")
        continue
    fi
    echo "launch top-k shard ${rank}/${NUM_SHARDS} on GPUs ${group}"
    CUDA_VISIBLE_DEVICES="${group}" python scripts/teacher_prefix/generate_teacher_prefix_topk.py \
        --input "${input}" \
        --output "${output}" \
        --teacher-model "${TOPK_TEACHER_MODEL}" \
        --tokenizer "${TOPK_TEACHER_MODEL}" \
        --top-k "${TOPK_K}" \
        --batch-size "${TOPK_BATCH_SIZE}" \
        --max-length "${TOPK_MAX_LENGTH}" \
        --dtype bfloat16 \
        --use-generated-token-ids \
        "${THINKING_ARG[@]}" >"${log}" 2>&1 &
    pids+=("$!")
done

while :; do
    running=0
    status=()
    for rank in "${!pids[@]}"; do
        pid=${pids[$rank]}
        output=${outputs[$rank]}
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            running=$((running + 1))
            status+=("shard${rank}=running")
        elif [ -f "${output}" ]; then
            status+=("shard${rank}=done")
        else
            status+=("shard${rank}=stopped")
        fi
    done
    echo "teacher-prefix top-k workers running: ${running}/${NUM_SHARDS}; ${status[*]}"
    [ "${running}" -eq 0 ] && break
    sleep 30
done

failed=0
for rank in "${!pids[@]}"; do
    pid=${pids[$rank]}
    if [ -n "${pid}" ] && ! wait "${pid}"; then
        echo "top-k shard ${rank} failed; tail of log:" >&2
        tail -80 "${SHARD_DIR}/$(printf 'topk_shard_%02d_of_%02d.log' "$rank" "${NUM_SHARDS}")" >&2
        failed=1
    fi
done
[ "${failed}" -eq 0 ] || exit 1

python scripts/teacher_prefix/merge_prefix_selection_shards.py \
    --inputs "${outputs[@]}" \
    --output "${TOPK_OUTPUT}"

python scripts/teacher_prefix/validate_teacher_prefix_topk.py \
    --input "${TOPK_INPUT}" \
    --output "${TOPK_OUTPUT}" \
    --top-k "${TOPK_K}"
