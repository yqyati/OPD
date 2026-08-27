#!/usr/bin/env bash
# Build a completion-aware fixed-prefix MOPD dataset from the retained 7k
# same-origin teacher trajectories. This script performs CPU data processing only.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl
source /mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/OPD/.env
cd "${OPD_ROOT}"

PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
PREFIX_LENGTH=${PREFIX_LENGTH:?PREFIX_LENGTH is required}
case "${PREFIX_LENGTH}" in
  128|256|512) ;;
  *) echo "Unsupported PREFIX_LENGTH=${PREFIX_LENGTH}; expected 128, 256, or 512" >&2; exit 2 ;;
esac

WORK_DIR=datasets/mopd/sameorigin_full7168_teacher_data
EFFECTIVE_DATASET="${WORK_DIR}/q8b_pragmatic_plain_mopd_effective_maxprompt2048.parquet"
OUTPUT=${PREFIX_OUTPUT:-datasets/mopd/q8b_sameorigin_mopd_math_code_instruct_teacher_prefix${PREFIX_LENGTH}_think.parquet}

for domain in math instruct code; do
  test -f "${WORK_DIR}/${domain}_prompts.parquet" || { echo "Missing ${domain} prompt split" >&2; exit 1; }
  test -f "${WORK_DIR}/${domain}_full_response_7168.parquet" || { echo "Missing ${domain} 7k trajectories" >&2; exit 1; }
done
test -f "${EFFECTIVE_DATASET}" || { echo "Missing effective manifest: ${EFFECTIVE_DATASET}" >&2; exit 1; }

if [[ ! -f "${OUTPUT}" ]]; then
  for domain in math instruct code; do
    prefix="${WORK_DIR}/${domain}_prefix${PREFIX_LENGTH}.parquet"
    if [[ ! -f "${prefix}" ]]; then
      "${PYTHON_BIN}" scripts/teacher_prefix/build_prefix_dataset_from_teacher_responses.py \
        --input "${WORK_DIR}/${domain}_full_response_7168.parquet" \
        --source "${WORK_DIR}/${domain}_prompts.parquet" \
        --output "${prefix}" \
        --prefix-length "${PREFIX_LENGTH}"
    else
      echo "Reusing ${prefix}"
    fi
  done

  WORK_DIR="${WORK_DIR}" EFFECTIVE_DATASET="${EFFECTIVE_DATASET}" OUTPUT="${OUTPUT}" PREFIX_LENGTH="${PREFIX_LENGTH}" \
    "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

work_dir = Path(os.environ["WORK_DIR"])
source_path = Path(os.environ["EFFECTIVE_DATASET"])
output_path = Path(os.environ["OUTPUT"])
prefix_length = int(os.environ["PREFIX_LENGTH"])
temporary = Path(f"{output_path}.writing")
if temporary.exists():
    temporary.unlink()

tables = [pq.read_table(work_dir / f"{domain}_prefix{prefix_length}.parquet") for domain in ("math", "instruct", "code")]
merged = pa.concat_tables(tables).sort_by([("__mopd_manifest_index", "ascending")])
merged = merged.drop(["__mopd_manifest_index"])
source = pq.read_table(source_path, columns=["mopd_domain", "mopd_source_row"])
if merged.num_rows != source.num_rows:
    raise RuntimeError(f"Merged row count changed: {merged.num_rows} != {source.num_rows}")
for column in ("mopd_domain", "mopd_source_row"):
    if not merged[column].equals(source[column]):
        raise RuntimeError(f"Merged row order differs from source in {column}")
pq.write_table(merged, temporary, compression="zstd")
temporary.replace(output_path)
print(f"Wrote {output_path}: rows={merged.num_rows}")
PY
else
  echo "Reusing ${OUTPUT}"
fi

OUTPUT="${OUTPUT}" PREFIX_LENGTH="${PREFIX_LENGTH}" \
MATH_TEACHER=merged_models/q8b_math_grpo_from_general_sft_ep1_r7168_n8_b48_ep1_shuffle42_lr5e-6_step373 \
INSTRUCT_TEACHER=merged_models/q8b_instruct_grpo_from_general_sft_ep1_r7168_n8_b48_ep1_shuffle42_lr5e-6_step516 \
CODE_TEACHER=merged_models/q8b_code_grpo_from_general_sft_ep1_r7168_n8_b48_ep1_shuffle42_lr5e-6_step523 \
  "${PYTHON_BIN}" - <<'PY'
import os
from collections import Counter

import pyarrow.parquet as pq

output_path = os.environ["OUTPUT"]
prefix_length = int(os.environ["PREFIX_LENGTH"])
expected_counts = {"math": 17917, "instruct": 24809, "code": 25110}
expected_models = {
    "math": os.environ["MATH_TEACHER"],
    "instruct": os.environ["INSTRUCT_TEACHER"],
    "code": os.environ["CODE_TEACHER"],
}
table = pq.read_table(
    output_path,
    columns=[
        "mopd_domain",
        "teacher_prefix_token_ids",
        "teacher_prefix_token_len",
        "teacher_prefix_finish_reason",
        "teacher_prefix_model",
        "teacher_prefix_enable_thinking",
    ],
)
if table.num_rows != 67836:
    raise RuntimeError(f"Unexpected final row count: {table.num_rows}")
rows = table.to_pylist()
counts = Counter(row["mopd_domain"] for row in rows)
if dict(counts) != expected_counts:
    raise RuntimeError(f"Domain counts changed: {dict(counts)}")
for index, row in enumerate(rows):
    domain = row["mopd_domain"]
    ids = row["teacher_prefix_token_ids"]
    length = row["teacher_prefix_token_len"]
    finish_reason = row["teacher_prefix_finish_reason"]
    if length != len(ids) or not 0 < length <= prefix_length:
        raise RuntimeError(f"Invalid prefix length at row {index}: {length}")
    if finish_reason == "length" and length != prefix_length:
        raise RuntimeError(f"Truncated prefix at row {index} has length {length}")
    if finish_reason not in {"length", "stop"}:
        raise RuntimeError(f"Unexpected finish reason at row {index}: {finish_reason}")
    if row["teacher_prefix_model"] != expected_models[domain]:
        raise RuntimeError(f"Wrong teacher at row {index}: {row['teacher_prefix_model']}")
    if not row["teacher_prefix_enable_thinking"]:
        raise RuntimeError(f"Thinking disabled at row {index}")
print(f"Validated Prefix-{prefix_length}: rows={table.num_rows}, domains={dict(counts)}")
PY

echo "Prefix-${PREFIX_LENGTH} dataset ready: ${OUTPUT}"
