#!/usr/bin/env bash
set -euo pipefail

: "${PREFIX_INPUT:?Set PREFIX_INPUT}"
: "${PREFIX_OUTPUT:?Set PREFIX_OUTPUT}"
: "${PREFIX_TEACHER_MODEL:?Set PREFIX_TEACHER_MODEL}"
: "${PREFIX_GPU_GROUPS:?Set PREFIX_GPU_GROUPS, e.g. '0;1;2;3'}"

PREFIX_TP=${PREFIX_TP:-1}
PREFIX_MAX_TOKENS=${PREFIX_MAX_TOKENS:-1024}
PREFIX_MAX_MODEL_LEN=${PREFIX_MAX_MODEL_LEN:-2048}
PREFIX_BATCH_SIZE=${PREFIX_BATCH_SIZE:-384}
PREFIX_TEMPERATURE=${PREFIX_TEMPERATURE:-0.7}
PREFIX_TOP_P=${PREFIX_TOP_P:-0.95}
PREFIX_ENABLE_THINKING=${PREFIX_ENABLE_THINKING:-True}

case "${PREFIX_ENABLE_THINKING,,}" in
    true) THINKING_ARG=(--enable-thinking) ;;
    false) THINKING_ARG=() ;;
    *) echo "PREFIX_ENABLE_THINKING must be True or False" >&2; exit 1 ;;
esac

if [ -f "${PREFIX_OUTPUT}" ]; then
    echo "Teacher-prefix dataset already exists; skipping Stage 1: ${PREFIX_OUTPUT}"
    exit 0
fi

IFS=';' read -r -a GPU_GROUPS <<< "${PREFIX_GPU_GROUPS}"
NUM_SHARDS=${#GPU_GROUPS[@]}
if [ "${NUM_SHARDS}" -lt 1 ]; then
    echo "No GPU groups configured" >&2
    exit 1
fi

SHARD_DIR="${PREFIX_OUTPUT%.parquet}_shards"
mkdir -p "${SHARD_DIR}"

PREFIX_INPUT="${PREFIX_INPUT}" SHARD_DIR="${SHARD_DIR}" NUM_SHARDS="${NUM_SHARDS}" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

src = os.environ["PREFIX_INPUT"]
out = Path(os.environ["SHARD_DIR"])
nshards = int(os.environ["NUM_SHARDS"])
df = pd.read_parquet(src).reset_index(drop=True)
df["__opd_original_index"] = range(len(df))
for rank in range(nshards):
    start = len(df) * rank // nshards
    end = len(df) * (rank + 1) // nshards
    path = out / f"input_shard_{rank:02d}_of_{nshards:02d}.parquet"
    if not path.exists():
        df.iloc[start:end].to_parquet(path, index=False)
    print(f"input shard {rank:02d}: rows={end-start} range=[{start},{end}) path={path}")
PY

pids=()
outputs=()
for rank in "${!GPU_GROUPS[@]}"; do
    group=${GPU_GROUPS[$rank]}
    input=$(printf '%s/input_shard_%02d_of_%02d.parquet' "${SHARD_DIR}" "${rank}" "${NUM_SHARDS}")
    output=$(printf '%s/prefix_shard_%02d_of_%02d.parquet' "${SHARD_DIR}" "${rank}" "${NUM_SHARDS}")
    log=$(printf '%s/prefix_shard_%02d_of_%02d.log' "${SHARD_DIR}" "${rank}" "${NUM_SHARDS}")
    outputs+=("${output}")
    echo "launch shard ${rank}/${NUM_SHARDS} on GPUs ${group}"
    python scripts/teacher_prefix/generate_teacher_prefix_data.py \
        --input "${input}" \
        --output "${output}" \
        --teacher-model "${PREFIX_TEACHER_MODEL}" \
        --gpus "${group}" \
        --tensor-parallel-size "${PREFIX_TP}" \
        --max-model-len "${PREFIX_MAX_MODEL_LEN}" \
        --max-new-tokens "${PREFIX_MAX_TOKENS}" \
        --temperature "${PREFIX_TEMPERATURE}" \
        --top-p "${PREFIX_TOP_P}" \
        --batch-size "${PREFIX_BATCH_SIZE}" \
        "${THINKING_ARG[@]}" >"${log}" 2>&1 &
    pids+=("$!")
done

while :; do
    running=0
    status=()
    for rank in "${!pids[@]}"; do
        if kill -0 "${pids[$rank]}" 2>/dev/null; then
            running=$((running + 1))
        fi
        tmp=$(printf '%s/prefix_shard_%02d_of_%02d.jsonl.tmp' "${SHARD_DIR}" "${rank}" "${NUM_SHARDS}")
        done_rows=0
        if [ -f "${tmp}" ]; then
            done_rows=$(wc -l < "${tmp}")
        fi
        status+=("shard${rank}=${done_rows}")
    done
    echo "prefix generation workers running: ${running}/${NUM_SHARDS}; ${status[*]}"
    [ "${running}" -eq 0 ] && break
    sleep 30
done

failed=0
for rank in "${!pids[@]}"; do
    if ! wait "${pids[$rank]}"; then
        echo "prefix shard ${rank} failed; tail of log:" >&2
        tail -80 "${SHARD_DIR}/$(printf 'prefix_shard_%02d_of_%02d.log' "${rank}" "${NUM_SHARDS}")" >&2
        failed=1
    fi
done
[ "${failed}" -eq 0 ] || exit 1

python scripts/teacher_prefix/merge_prefix_selection_shards.py \
    --inputs "${outputs[@]}" \
    --output "${PREFIX_OUTPUT}"

PREFIX_OUTPUT="${PREFIX_OUTPUT}" PREFIX_INPUT="${PREFIX_INPUT}" python - <<'PY'
import os
import pandas as pd

src = pd.read_parquet(os.environ["PREFIX_INPUT"])
out = pd.read_parquet(os.environ["PREFIX_OUTPUT"])
if len(src) != len(out):
    raise RuntimeError(f"row count mismatch: input={len(src)} output={len(out)}")
required = {"teacher_prefix_text", "teacher_prefix_token_ids", "teacher_prefix_enable_thinking"}
missing = required.difference(out.columns)
if missing:
    raise RuntimeError(f"missing required columns: {sorted(missing)}")
if not out["teacher_prefix_enable_thinking"].all():
    raise RuntimeError("generated dataset contains non-thinking prefixes")
print(f"validated teacher-prefix dataset: rows={len(out)} output={os.environ['PREFIX_OUTPUT']}")
PY
