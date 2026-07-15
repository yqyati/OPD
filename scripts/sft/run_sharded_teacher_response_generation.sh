#!/usr/bin/env bash
set -euo pipefail

: "${RESPONSE_INPUT:?Set RESPONSE_INPUT}"
: "${RESPONSE_OUTPUT:?Set RESPONSE_OUTPUT}"
: "${RESPONSE_TEACHER_MODEL:?Set RESPONSE_TEACHER_MODEL}"
: "${RESPONSE_GPU_GROUPS:?Set RESPONSE_GPU_GROUPS, e.g. '0;1;2;3'}"

RESPONSE_TP=${RESPONSE_TP:-1}
RESPONSE_MAX_TOKENS=${RESPONSE_MAX_TOKENS:-7168}
RESPONSE_MAX_MODEL_LEN=${RESPONSE_MAX_MODEL_LEN:-10240}
RESPONSE_BATCH_SIZE=${RESPONSE_BATCH_SIZE:-256}
RESPONSE_TEMPERATURE=${RESPONSE_TEMPERATURE:-0.7}
RESPONSE_TOP_P=${RESPONSE_TOP_P:-0.95}
RESPONSE_ENABLE_THINKING=${RESPONSE_ENABLE_THINKING:-True}

case "${RESPONSE_ENABLE_THINKING,,}" in
    true) THINKING_ARG=(--enable-thinking) ;;
    false) THINKING_ARG=() ;;
    *) echo "RESPONSE_ENABLE_THINKING must be True or False" >&2; exit 1 ;;
esac

if [ -f "${RESPONSE_OUTPUT}" ]; then
    echo "Teacher-response dataset already exists; skipping Stage 1: ${RESPONSE_OUTPUT}"
    exit 0
fi

IFS=';' read -r -a GPU_GROUPS <<< "${RESPONSE_GPU_GROUPS}"
NUM_SHARDS=${#GPU_GROUPS[@]}
if [ "${NUM_SHARDS}" -lt 1 ]; then
    echo "No GPU groups configured" >&2
    exit 1
fi

SHARD_DIR="${RESPONSE_OUTPUT%.parquet}_shards"
mkdir -p "${SHARD_DIR}"

RESPONSE_INPUT="${RESPONSE_INPUT}" SHARD_DIR="${SHARD_DIR}" NUM_SHARDS="${NUM_SHARDS}" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

src = os.environ["RESPONSE_INPUT"]
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
    output=$(printf '%s/response_shard_%02d_of_%02d.parquet' "${SHARD_DIR}" "${rank}" "${NUM_SHARDS}")
    log=$(printf '%s/response_shard_%02d_of_%02d.log' "${SHARD_DIR}" "${rank}" "${NUM_SHARDS}")
    outputs+=("${output}")
    echo "launch response shard ${rank}/${NUM_SHARDS} on GPUs ${group}"
    python scripts/sft/generate_teacher_response_data.py \
        --input "${input}" \
        --output "${output}" \
        --teacher-model "${RESPONSE_TEACHER_MODEL}" \
        --gpus "${group}" \
        --tensor-parallel-size "${RESPONSE_TP}" \
        --max-model-len "${RESPONSE_MAX_MODEL_LEN}" \
        --max-new-tokens "${RESPONSE_MAX_TOKENS}" \
        --temperature "${RESPONSE_TEMPERATURE}" \
        --top-p "${RESPONSE_TOP_P}" \
        --batch-size "${RESPONSE_BATCH_SIZE}" \
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
        tmp=$(printf '%s/response_shard_%02d_of_%02d.jsonl.tmp' "${SHARD_DIR}" "${rank}" "${NUM_SHARDS}")
        done_rows=0
        if [ -f "${tmp}" ]; then
            done_rows=$(wc -l < "${tmp}")
        fi
        status+=("shard${rank}=${done_rows}")
    done
    echo "teacher-response workers running: ${running}/${NUM_SHARDS}; ${status[*]}"
    [ "${running}" -eq 0 ] && break
    sleep 30
done

failed=0
for rank in "${!pids[@]}"; do
    if ! wait "${pids[$rank]}"; then
        echo "response shard ${rank} failed; tail of log:" >&2
        tail -80 "${SHARD_DIR}/$(printf 'response_shard_%02d_of_%02d.log' "${rank}" "${NUM_SHARDS}")" >&2
        failed=1
    fi
done
[ "${failed}" -eq 0 ] || exit 1

python scripts/teacher_prefix/merge_prefix_selection_shards.py \
    --inputs "${outputs[@]}" \
    --output "${RESPONSE_OUTPUT}"

RESPONSE_OUTPUT="${RESPONSE_OUTPUT}" RESPONSE_INPUT="${RESPONSE_INPUT}" python - <<'PY'
import os
import pandas as pd

src = pd.read_parquet(os.environ["RESPONSE_INPUT"])
out = pd.read_parquet(os.environ["RESPONSE_OUTPUT"])
if len(src) != len(out):
    raise RuntimeError(f"row count mismatch: input={len(src)} output={len(out)}")
required = {"teacher_response_text", "teacher_response_token_ids", "teacher_response_enable_thinking"}
missing = required.difference(out.columns)
if missing:
    raise RuntimeError(f"missing required columns: {sorted(missing)}")
if not out["teacher_response_enable_thinking"].all():
    raise RuntimeError("generated dataset contains non-thinking teacher responses")
if any(len(ids) == 0 for ids in out["teacher_response_token_ids"]):
    raise RuntimeError("generated dataset contains empty teacher response token IDs")
print(f"validated teacher-response dataset: rows={len(out)} output={os.environ['RESPONSE_OUTPUT']}")
PY
