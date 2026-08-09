#!/usr/bin/env bash
# Build the exact completion-aware 1024-token math handoff dataset from the
# already saved 7k no-think teacher trajectories.  No teacher rollout is
# performed here, and this script never submits an rjob.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

FULL_RESPONSE_DATASET="${FULL_RESPONSE_DATASET:-datasets/sft_teacher_response/q4binst_q30binst2507_nothink_full_response_7168.parquet}"
SOURCE_DATASET="${SOURCE_DATASET:-datasets/dapo-math-17k-teacher-aligned.parquet}"
OUTPUT_DATASET="${OUTPUT_DATASET:-datasets/teacher_prefix/q4binst_q30binst2507_nothink_dapo_math17k_prefix1024.parquet}"

test -f "${FULL_RESPONSE_DATASET}" || {
    echo "Missing saved 7k teacher trajectories: ${FULL_RESPONSE_DATASET}" >&2
    exit 1
}

if [ ! -f "${OUTPUT_DATASET}" ]; then
    python scripts/teacher_prefix/build_prefix_dataset_from_teacher_responses.py \
        --input "${FULL_RESPONSE_DATASET}" \
        --source "${SOURCE_DATASET}" \
        --output "${OUTPUT_DATASET}" \
        --prefix-length 1024
else
    echo "Reusing existing fixed-prefix dataset: ${OUTPUT_DATASET}"
fi

python - "${OUTPUT_DATASET}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_parquet(path)
required = {
    "teacher_prefix_token_ids", "teacher_prefix_token_len",
    "teacher_prefix_finish_reason", "teacher_prefix_enable_thinking",
}
missing = required.difference(df.columns)
if missing:
    raise RuntimeError(f"Missing prefix columns: {sorted(missing)}")
if len(df) != 17917:
    raise RuntimeError(f"Unexpected row count {len(df)}; expected 17917")
if not df.teacher_prefix_token_len.between(1, 1024).all():
    raise RuntimeError("Invalid prefix lengths")
if df.teacher_prefix_enable_thinking.astype(bool).any():
    raise RuntimeError("Prefix data unexpectedly has thinking enabled")
stopped = df.teacher_prefix_finish_reason.eq("stop")
rollout = df.teacher_prefix_finish_reason.eq("length")
if not (stopped | rollout).all():
    raise RuntimeError("Unexpected teacher prefix finish reason")
if not (df.loc[rollout, "teacher_prefix_token_len"] == 1024).all():
    raise RuntimeError("Every suffix-OPD row must contain exactly 1024 prefix tokens")
print(
    f"Validated {path}: rows={len(df)}, suffix_opd_rows={int(rollout.sum())}, "
    f"short_complete_sft_only_rows={int(stopped.sum())}"
)
PY
