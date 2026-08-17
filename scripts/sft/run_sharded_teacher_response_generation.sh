#!/usr/bin/env bash
set -euo pipefail

: "${RESPONSE_INPUT:?Set RESPONSE_INPUT}"
: "${RESPONSE_OUTPUT:?Set RESPONSE_OUTPUT}"
: "${RESPONSE_TEACHER_MODEL:?Set RESPONSE_TEACHER_MODEL}"
: "${RESPONSE_GPU_GROUPS:?Set RESPONSE_GPU_GROUPS, e.g. '0;1;2;3'}"

RESPONSE_TP=${RESPONSE_TP:-1}
RESPONSE_MAX_TOKENS=${RESPONSE_MAX_TOKENS:-7168}
RESPONSE_MAX_MODEL_LEN=${RESPONSE_MAX_MODEL_LEN:-10240}
RESPONSE_BATCH_SIZE=${RESPONSE_BATCH_SIZE:-128}
RESPONSE_TEMPERATURE=${RESPONSE_TEMPERATURE:-0.7}
RESPONSE_TOP_P=${RESPONSE_TOP_P:-0.95}
RESPONSE_ENABLE_THINKING=${RESPONSE_ENABLE_THINKING:-True}
RESPONSE_SKIP_OVERLONG_PROMPTS=${RESPONSE_SKIP_OVERLONG_PROMPTS:-False}

case "${RESPONSE_ENABLE_THINKING,,}" in
    true) THINKING_ARG=(--enable-thinking) ;;
    false) THINKING_ARG=() ;;
    *) echo "RESPONSE_ENABLE_THINKING must be True or False" >&2; exit 1 ;;
esac

case "${RESPONSE_SKIP_OVERLONG_PROMPTS,,}" in
    true) SKIP_OVERLONG_ARG=(--skip-overlong-prompts) ;;
    false) SKIP_OVERLONG_ARG=() ;;
    *) echo "RESPONSE_SKIP_OVERLONG_PROMPTS must be True or False" >&2; exit 1 ;;
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
        "${SKIP_OVERLONG_ARG[@]}" \
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

# Rebuild each shard from the resumable JSONL artifacts.  This is deliberately
# done even when a stale parquet shard exists: older generator versions could
# write the text columns while dropping the nested token-id list, which would
# make the final artifact unusable for SFT/prefix selection.
RESPONSE_INPUT="${RESPONSE_INPUT}" SHARD_DIR="${SHARD_DIR}" NUM_SHARDS="${NUM_SHARDS}" python - <<'PY'
import json, os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

src = Path(os.environ["RESPONSE_INPUT"])
shard_dir = Path(os.environ["SHARD_DIR"])
n = int(os.environ["NUM_SHARDS"])
response_columns = [
    "teacher_response_text",
    "teacher_response_token_ids",
    "teacher_response_token_len",
    "teacher_response_finish_reason",
    "teacher_response_status",
    "teacher_response_model",
    "teacher_response_max_tokens",
    "teacher_response_temperature",
    "teacher_response_top_p",
    "teacher_response_enable_thinking",
]
for rank in range(n):
    inp = shard_dir / f"input_shard_{rank:02d}_of_{n:02d}.parquet"
    js = shard_dir / f"response_shard_{rank:02d}_of_{n:02d}.jsonl.tmp"
    out = shard_dir / f"response_shard_{rank:02d}_of_{n:02d}.parquet"
    records = {}
    with js.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec.setdefault("teacher_response_status", "generated")
            rec.setdefault("teacher_response_prompt_token_len", None)
            records[int(rec["__teacher_response_row_id"])] = rec
    source = pq.read_table(inp)
    rows = []
    for i in range(source.num_rows):
        if i not in records:
            raise RuntimeError(f"missing JSONL row {i} in {js}")
        rec = records[i]
        rows.append({key: rec.get(key) for key in response_columns})
    response = pa.Table.from_pylist(rows)
    # Keep source columns and append response columns; avoid pandas/object
    # coercion so List<int64> token IDs remain intact.
    names = list(source.column_names) + list(response.column_names)
    table = pa.Table.from_arrays(list(source.columns) + list(response.columns), names=names)
    pq.write_table(table, out, compression="zstd")
    print(f"rebuilt {out} rows={table.num_rows} token_ids={('teacher_response_token_ids' in table.column_names)}")
PY

# Merge directly from JSONL as a final safeguard. The response parquet files
# are retained for debugging, but JSONL is the source of truth for nested token
# IDs because some pyarrow/pandas combinations silently omit that column.
# Input shards are contiguous source ranges in rank order, so write each in
# that order. Do not concatenate all shards: 25k long trajectories can exceed
# Arrow's 32-bit list/string offset limit during concat/take.
RESPONSE_INPUT="${RESPONSE_INPUT}" SHARD_DIR="${SHARD_DIR}" NUM_SHARDS="${NUM_SHARDS}" RESPONSE_OUTPUT="${RESPONSE_OUTPUT}" python - <<'PY'
import json, os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

d = Path(os.environ["SHARD_DIR"]); n = int(os.environ["NUM_SHARDS"])
output = Path(os.environ["RESPONSE_OUTPUT"])
temporary = output.with_suffix(output.suffix + ".writing")
if temporary.exists():
    temporary.unlink()
writer = None
total = 0
response_columns = [
    "teacher_response_text",
    "teacher_response_token_ids",
    "teacher_response_token_len",
    "teacher_response_finish_reason",
    "teacher_response_status",
    "teacher_response_model",
    "teacher_response_max_tokens",
    "teacher_response_temperature",
    "teacher_response_top_p",
    "teacher_response_enable_thinking",
]
for rank in range(n):
    inp = d / f"input_shard_{rank:02d}_of_{n:02d}.parquet"
    js = d / f"response_shard_{rank:02d}_of_{n:02d}.jsonl.tmp"
    expected_id = 0
    shard_rows = 0
    with js.open(encoding="utf-8") as f:
        for batch in pq.ParquetFile(inp).iter_batches(batch_size=64):
            source = pa.Table.from_batches([batch])
            records = []
            for _ in range(source.num_rows):
                line = f.readline()
                if not line:
                    raise RuntimeError(f"unexpected end of {js}")
                rec = json.loads(line)
                if rec.get("__teacher_response_row_id") != expected_id:
                    raise RuntimeError(f"out-of-order/missing response {expected_id} in {js}")
                expected_id += 1
                # JSONL can span generator versions. Keep every output batch
                # schema-identical before creating the streaming Parquet writer.
                rec.setdefault("teacher_response_status", "generated")
                rec.setdefault("teacher_response_prompt_token_len", None)
                records.append({key: rec.get(key) for key in response_columns})
            response = pa.Table.from_pylist(records)
            table = pa.Table.from_arrays(
                list(source.columns) + list(response.columns),
                names=list(source.column_names) + list(response.column_names),
            ).drop(["__opd_original_index"])
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            shard_rows += table.num_rows
            total += table.num_rows
        if f.readline():
            raise RuntimeError(f"extra records in {js}")
    print(f"merged shard {rank + 1}/{n}: rows={shard_rows}")
if writer is None:
    raise RuntimeError("No response shards to merge")
writer.close()
check = pq.ParquetFile(temporary)
if check.metadata.num_rows != total or "teacher_response_token_ids" not in check.schema_arrow.names:
    raise RuntimeError("Merged parquet validation failed")
temporary.replace(output)
print(f"Wrote {output} ({total} rows), token_ids=True")
PY

RESPONSE_OUTPUT="${RESPONSE_OUTPUT}" RESPONSE_INPUT="${RESPONSE_INPUT}" python - <<'PY'
import os
import pyarrow.parquet as pq

src = pq.ParquetFile(os.environ["RESPONSE_INPUT"])
out = pq.ParquetFile(os.environ["RESPONSE_OUTPUT"])
if src.metadata.num_rows != out.metadata.num_rows:
    raise RuntimeError(f"row count mismatch: input={src.metadata.num_rows} output={out.metadata.num_rows}")
required = {"teacher_response_text", "teacher_response_token_ids", "teacher_response_enable_thinking"}
# ParquetFile.schema.names is the physical leaf schema in some pyarrow
# versions and omits nested/list columns.  Use the Arrow schema's top-level
# names for validation instead.
top_level_names = set(out.schema_arrow.names)
missing = required.difference(top_level_names)
if missing:
    raise RuntimeError(f"missing required columns: {sorted(missing)}")
expected_thinking = os.environ["RESPONSE_ENABLE_THINKING"].lower() == "true"
columns = ["teacher_response_enable_thinking", "teacher_response_token_ids"]
if "teacher_response_status" in top_level_names:
    columns.append("teacher_response_status")
out_values = out.read(columns=columns)
thinking_values = out_values["teacher_response_enable_thinking"].to_pylist()
if not all(value == expected_thinking for value in thinking_values):
    raise RuntimeError(
        "teacher-response thinking mode does not match RESPONSE_ENABLE_THINKING="
        f"{expected_thinking}"
    )
statuses = (
    out_values["teacher_response_status"].to_pylist()
    if "teacher_response_status" in top_level_names
    else ["generated"] * out.metadata.num_rows
)
for status, ids in zip(statuses, out_values["teacher_response_token_ids"].to_pylist(), strict=True):
    if status in (None, "generated") and len(ids) == 0:
        raise RuntimeError("generated dataset contains empty teacher response token IDs")
print(f"validated teacher-response dataset: rows={out.metadata.num_rows} output={os.environ['RESPONSE_OUTPUT']}")
PY
