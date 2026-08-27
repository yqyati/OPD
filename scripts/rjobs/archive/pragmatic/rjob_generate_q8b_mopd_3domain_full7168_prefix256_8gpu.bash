#!/usr/bin/env bash
# Generate reusable 7k trajectories from the three MOPD specialist teachers,
# then slice a completion-aware fixed-256 prefix dataset for MOPD training.
# Run this script inside one 8-GPU H200 job. It does not submit a job itself.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl

export YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
export OPD_ROOT="${YANGQINGYU_ROOT}/workspace/OPD"
export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
cd "${OPD_ROOT}"

PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
SOURCE_DATASET=datasets/mopd/q8b_pragmatic_plain_mopd_math_code_instruct_prompts.parquet
WORK_DIR=datasets/mopd/prefix256_teacher_data
EFFECTIVE_DATASET="${WORK_DIR}/q8b_pragmatic_plain_mopd_effective_maxprompt2048.parquet"
FINAL_PREFIX_DATASET=datasets/mopd/q8b_mopd_math_code_instruct_teacher_prefix256_think.parquet

MATH_TEACHER=merged_models/q8b_q30ba3b_dapo_math17k_think_correct6108_max7168_sftinit_math_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step373
INSTRUCT_TEACHER=merged_models/q8b_q30ba3b_ifrlvr_think_correct5917_sftinit_ifrlvr_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step516
CODE_TEACHER=merged_models/q8b_q30ba3b_eurus_code_think_correct_max7168_sftinit_code_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step523

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

if [ "${MOPD_PREFIX_PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "MOPD prefix-generation preflight passed; GPU generation was not started."
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

if [ ! -f "${FINAL_PREFIX_DATASET}" ]; then
    run_domain_generation math "${MATH_TEACHER}" '0;1' &
    math_pid=$!
    run_domain_generation instruct "${INSTRUCT_TEACHER}" '2;3;4' &
    instruct_pid=$!
    run_domain_generation code "${CODE_TEACHER}" '5;6;7' &
    code_pid=$!

    failed=0
    wait "${math_pid}" || { echo "Math generation failed; inspect ${WORK_DIR}/math_generation_driver.log" >&2; failed=1; }
    wait "${instruct_pid}" || { echo "Instruction generation failed; inspect ${WORK_DIR}/instruct_generation_driver.log" >&2; failed=1; }
    wait "${code_pid}" || { echo "Code generation failed; inspect ${WORK_DIR}/code_generation_driver.log" >&2; failed=1; }
    [ "${failed}" -eq 0 ] || exit 1

    for domain in math instruct code; do
        prefix="${WORK_DIR}/${domain}_prefix256.parquet"
        if [ ! -f "${prefix}" ]; then
            "${PYTHON_BIN}" scripts/teacher_prefix/build_prefix_dataset_from_teacher_responses.py \
                --input "${WORK_DIR}/${domain}_full_response_7168.parquet" \
                --source "${WORK_DIR}/${domain}_prompts.parquet" \
                --output "${prefix}" \
                --prefix-length 256
        else
            echo "Reusing ${prefix}"
        fi
    done

    WORK_DIR="${WORK_DIR}" EFFECTIVE_DATASET="${EFFECTIVE_DATASET}" FINAL_PREFIX_DATASET="${FINAL_PREFIX_DATASET}" \
        "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

work_dir = Path(os.environ["WORK_DIR"])
source_path = os.environ["EFFECTIVE_DATASET"]
output_path = Path(os.environ["FINAL_PREFIX_DATASET"])
temporary = Path(f"{output_path}.writing")
tables = [pq.read_table(work_dir / f"{domain}_prefix256.parquet") for domain in ("math", "instruct", "code")]
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
    echo "Reusing final prefix dataset: ${FINAL_PREFIX_DATASET}"
fi

EFFECTIVE_DATASET="${EFFECTIVE_DATASET}" FINAL_PREFIX_DATASET="${FINAL_PREFIX_DATASET}" \
MATH_TEACHER="${MATH_TEACHER}" INSTRUCT_TEACHER="${INSTRUCT_TEACHER}" CODE_TEACHER="${CODE_TEACHER}" \
    "${PYTHON_BIN}" - <<'PY'
import os

import pyarrow.compute as pc
import pyarrow.parquet as pq

source = pq.read_table(os.environ["EFFECTIVE_DATASET"], columns=["mopd_domain", "mopd_source_row"])
output = pq.read_table(
    os.environ["FINAL_PREFIX_DATASET"],
    columns=[
        "mopd_domain",
        "mopd_source_row",
        "teacher_prefix_token_ids",
        "teacher_prefix_token_len",
        "teacher_prefix_finish_reason",
        "teacher_prefix_model",
        "teacher_prefix_enable_thinking",
    ],
)
if output.num_rows != 67836:
    raise RuntimeError(f"Unexpected final row count: {output.num_rows}")
for column in ("mopd_domain", "mopd_source_row"):
    if not output[column].equals(source[column]):
        raise RuntimeError(f"Final dataset order differs from source in {column}")
expected_counts = {"math": 17917, "instruct": 24809, "code": 25110}
expected_models = {
    "math": os.environ["MATH_TEACHER"],
    "instruct": os.environ["INSTRUCT_TEACHER"],
    "code": os.environ["CODE_TEACHER"],
}
rows = output.select(
    [
        "mopd_domain",
        "teacher_prefix_token_ids",
        "teacher_prefix_token_len",
        "teacher_prefix_finish_reason",
        "teacher_prefix_model",
        "teacher_prefix_enable_thinking",
    ]
).to_pylist()
counts = {domain: 0 for domain in expected_counts}
rollout_rows = 0
complete_rows = 0
for index, row in enumerate(rows):
    domain = row["mopd_domain"]
    counts[domain] += 1
    token_ids = row["teacher_prefix_token_ids"]
    token_len = row["teacher_prefix_token_len"]
    finish_reason = row["teacher_prefix_finish_reason"]
    if token_len != len(token_ids) or not 0 < token_len <= 256:
        raise RuntimeError(f"Invalid prefix length at row {index}: {token_len}")
    if finish_reason == "length":
        if token_len != 256:
            raise RuntimeError(f"Suffix-OPD row {index} does not have 256 tokens")
        rollout_rows += 1
    elif finish_reason == "stop":
        complete_rows += 1
    else:
        raise RuntimeError(f"Unexpected finish reason at row {index}: {finish_reason}")
    if row["teacher_prefix_model"] != expected_models[domain]:
        raise RuntimeError(f"Wrong teacher for {domain} at row {index}: {row['teacher_prefix_model']}")
    if not row["teacher_prefix_enable_thinking"]:
        raise RuntimeError(f"Thinking disabled at row {index}")
if counts != expected_counts:
    raise RuntimeError(f"Domain counts changed: {counts}")
print(
    f"Validated {os.environ['FINAL_PREFIX_DATASET']}: rows={output.num_rows}, "
    f"domains={counts}, suffix_opd_rows={rollout_rows}, complete_sft_only_rows={complete_rows}"
)
PY

echo "Full teacher responses: ${WORK_DIR}/{math,instruct,code}_full_response_7168.parquet"
echo "Prefix256 MOPD dataset: ${FINAL_PREFIX_DATASET}"
