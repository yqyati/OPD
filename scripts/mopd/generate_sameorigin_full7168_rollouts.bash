#!/usr/bin/env bash
# Shared implementation for generating reusable 7k trajectories from the
# three same-origin MOPD teachers. Call it through a concrete pipeline launcher.
# Run this script inside one 8-GPU H200 job. It does not submit a job itself.
set -euo pipefail

# This implementation owns rollout only. Prefix launchers consume its outputs
# after this process exits; fixed-prefix construction is not executed here.
MOPD_FULL_RESPONSE_ONLY=1

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl
source /mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/OPD/.env

export YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
export OPD_ROOT="${YANGQINGYU_ROOT}/workspace/OPD"
export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
cd "${OPD_ROOT}"

PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
SOURCE_DATASET=datasets/mopd/q8b_pragmatic_plain_mopd_math_code_instruct_prompts.parquet
WORK_DIR=${MOPD_TEACHER_DATA_DIR:?MOPD_TEACHER_DATA_DIR must select the rollout output directory}
EFFECTIVE_DATASET="${WORK_DIR}/q8b_pragmatic_plain_mopd_effective_maxprompt2048.parquet"

MATH_TEACHER=merged_models/q8b_math_grpo_from_general_sft_ep1_r7168_n8_b48_ep1_shuffle42_lr5e-6_step373
INSTRUCT_TEACHER=merged_models/q8b_instruct_grpo_from_general_sft_ep1_r7168_n8_b48_ep1_shuffle42_lr5e-6_step516
CODE_TEACHER=merged_models/q8b_code_grpo_from_general_sft_ep1_r7168_n8_b48_ep1_shuffle42_lr5e-6_step523

mkdir -p "${WORK_DIR}"
for required in \
    "${SOURCE_DATASET}" \
    "${MATH_TEACHER}/config.json" \
    "${INSTRUCT_TEACHER}/config.json" \
    "${CODE_TEACHER}/config.json"; do
    test -f "${required}" || { echo "Missing required file: ${required}" >&2; exit 1; }
done

SOURCE_DATASET="${SOURCE_DATASET}" EFFECTIVE_DATASET="${EFFECTIVE_DATASET}" WORK_DIR="${WORK_DIR}" \
MATH_TEACHER="${MATH_TEACHER}" INSTRUCT_TEACHER="${INSTRUCT_TEACHER}" CODE_TEACHER="${CODE_TEACHER}" \
    "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import AutoTokenizer

source = os.environ["SOURCE_DATASET"]
effective_path = Path(os.environ["EFFECTIVE_DATASET"])
work_dir = Path(os.environ["WORK_DIR"])
table = pq.read_table(source)
source_counts = {"math": 17917, "instruct": 25056, "code": 25110}
expected = {"math": 17917, "instruct": 24809, "code": 25110}
teachers = {
    "math": os.environ["MATH_TEACHER"],
    "instruct": os.environ["INSTRUCT_TEACHER"],
    "code": os.environ["CODE_TEACHER"],
}
if table.num_rows != sum(source_counts.values()):
    raise RuntimeError(f"Unexpected MOPD row count: {table.num_rows}")
manifest_index = pa.array(range(table.num_rows), type=pa.int64())
table = table.append_column("__mopd_manifest_index", manifest_index)

# Match RLHFDataset(filter_overlong_prompts=True, max_prompt_length=2048)
# before spending GPU time. The original source manifest remains untouched.
kept_tables = []
for domain, source_rows in source_counts.items():
    subset = table.filter(pc.equal(table["mopd_domain"], domain))
    if subset.num_rows != source_rows:
        raise RuntimeError(f"Unexpected source {domain} rows: {subset.num_rows}; expected {source_rows}")
    tokenizer = AutoTokenizer.from_pretrained(teachers[domain], trust_remote_code=True)
    keep = []
    max_seen = 0
    for prompt in subset["prompt"].to_pylist():
        messages = [dict(message) for message in prompt]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_len = len(tokenizer.encode(rendered, add_special_tokens=False))
        max_seen = max(max_seen, prompt_len)
        keep.append(prompt_len <= 2048)
    subset = subset.filter(pa.array(keep, type=pa.bool_()))
    if subset.num_rows != expected[domain]:
        raise RuntimeError(
            f"Unexpected effective {domain} rows: {subset.num_rows}; expected {expected[domain]}"
        )
    print(
        f"Filtered {domain}: source={source_rows}, kept={subset.num_rows}, "
        f"overlong={source_rows - subset.num_rows}, max_prompt_tokens={max_seen}"
    )
    kept_tables.append(subset)

effective = pa.concat_tables(kept_tables).sort_by([("__mopd_manifest_index", "ascending")])
pq.write_table(effective, effective_path, compression="zstd")
print(f"Wrote effective MOPD manifest: {effective_path}, rows={effective.num_rows}")

for domain, expected_rows in expected.items():
    output = work_dir / f"{domain}_prompts.parquet"
    subset = effective.filter(pc.equal(effective["mopd_domain"], domain))
    if subset.num_rows != expected_rows:
        raise RuntimeError(f"Unexpected {domain} rows: {subset.num_rows}; expected {expected_rows}")
    if output.exists():
        existing = pq.ParquetFile(output)
        if existing.metadata.num_rows != expected_rows:
            raise RuntimeError(
                f"Stale domain input has wrong row count: {output}. "
                "Remove only this stale prompt split and rerun."
            )
        print(f"Reusing {output}: rows={expected_rows}")
    else:
        pq.write_table(subset, output, compression="zstd")
        print(f"Wrote {output}: rows={expected_rows}")
PY

if [ "${MOPD_ROLLOUT_PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "MOPD rollout preflight passed; GPU generation was not started."
    exit 0
fi

run_domain_generation() {
    local domain=$1
    local teacher=$2
    local gpu_groups=$3
    local input="${WORK_DIR}/${domain}_prompts.parquet"
    local responses="${WORK_DIR}/${domain}_full_response_7168.parquet"
    local log="${WORK_DIR}/${domain}_generation_driver.log"

    echo "Starting ${domain}: teacher=${teacher}, GPUs=${gpu_groups}"
    RESPONSE_INPUT="${input}" \
    RESPONSE_OUTPUT="${responses}" \
    RESPONSE_TEACHER_MODEL="${teacher}" \
    RESPONSE_GPU_GROUPS="${gpu_groups}" \
    RESPONSE_TP=1 \
    RESPONSE_MAX_TOKENS=7168 \
    RESPONSE_MAX_MODEL_LEN=9216 \
    RESPONSE_BATCH_SIZE=64 \
    RESPONSE_TEMPERATURE=0.7 \
    RESPONSE_TOP_P=0.95 \
    RESPONSE_ENABLE_THINKING=True \
    bash scripts/sft/run_sharded_teacher_response_generation.sh >"${log}" 2>&1
}

generation_pids=()
generation_domains=()
generation_logs=()
if [ ! -f "${WORK_DIR}/math_full_response_7168.parquet" ]; then
    run_domain_generation math "${MATH_TEACHER}" '0;1' &
    generation_pids+=("$!")
    generation_domains+=(math)
    generation_logs+=("${WORK_DIR}/math_generation_driver.log")
else
    echo "Reusing ${WORK_DIR}/math_full_response_7168.parquet"
fi
if [ ! -f "${WORK_DIR}/instruct_full_response_7168.parquet" ]; then
    run_domain_generation instruct "${INSTRUCT_TEACHER}" '2;3;4' &
    generation_pids+=("$!")
    generation_domains+=(instruct)
    generation_logs+=("${WORK_DIR}/instruct_generation_driver.log")
else
    echo "Reusing ${WORK_DIR}/instruct_full_response_7168.parquet"
fi
if [ ! -f "${WORK_DIR}/code_full_response_7168.parquet" ]; then
    run_domain_generation code "${CODE_TEACHER}" '5;6;7' &
    generation_pids+=("$!")
    generation_domains+=(code)
    generation_logs+=("${WORK_DIR}/code_generation_driver.log")
else
    echo "Reusing ${WORK_DIR}/code_full_response_7168.parquet"
fi

failed=0
for index in "${!generation_pids[@]}"; do
    if ! wait "${generation_pids[$index]}"; then
        echo "${generation_domains[$index]} generation failed; inspect ${generation_logs[$index]}" >&2
        failed=1
    fi
done
[ "${failed}" -eq 0 ] || exit 1

for domain in math instruct code; do
    response="${WORK_DIR}/${domain}_full_response_7168.parquet"
    test -f "${response}" || { echo "Missing teacher-response output: ${response}" >&2; exit 1; }
done

if [ "${MOPD_FULL_RESPONSE_ONLY:-0}" = "1" ]; then
    WORK_DIR="${WORK_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pyarrow.parquet as pq

work_dir = Path(os.environ["WORK_DIR"])
expected = {"math": 17917, "instruct": 24809, "code": 25110}
required_columns = {
    "teacher_response_text",
    "teacher_response_token_ids",
    "teacher_response_finish_reason",
    "teacher_response_model",
    "teacher_response_enable_thinking",
}
for domain, expected_rows in expected.items():
    path = work_dir / f"{domain}_full_response_7168.parquet"
    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata.num_rows != expected_rows:
        raise RuntimeError(
            f"{path}: expected {expected_rows} rows, got {parquet_file.metadata.num_rows}"
        )
    missing = required_columns.difference(parquet_file.schema_arrow.names)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    print(f"Validated reusable 7k teacher trajectories: {path} ({expected_rows} rows)")
PY
    echo "Full-response rollout generation complete."
    exit 0
fi
