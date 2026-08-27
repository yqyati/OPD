#!/usr/bin/env bash
# Qwen3-8B-Base -> full-response three-domain SFT -> evaluation
# -> SFT-initialized plain MOPD -> evaluation.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl
source /mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/OPD/.env

export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_TIMEOUT=7200
cd "${OPD_ROOT}"

PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
N_GPUS_PER_NODE=8
STUDENT_MODEL="${MODEL_ROOT}/Qwen3-8B-Base"

MATH_SOURCE=datasets/mopd/prefix256_teacher_data/math_full_response_7168.parquet
CODE_SOURCE=datasets/mopd/prefix256_teacher_data/code_full_response_7168.parquet
INSTRUCT_SOURCE=datasets/mopd/prefix256_teacher_data/instruct_full_response_7168.parquet
FULL_RESPONSE_SOURCE=datasets/mopd/q8b_mopd_math_code_instruct_full_response_7168.parquet
SFT_DATASET=datasets/sft/q8b_mopd_math_code_instruct_full7168_exact_tokens_sft.parquet

MATH_TEACHER=merged_models/q8b_q30ba3b_dapo_math17k_think_correct6108_max7168_sftinit_math_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step373
CODE_TEACHER=merged_models/q8b_q30ba3b_eurus_code_think_correct_max7168_sftinit_code_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step523
INSTRUCT_TEACHER=merged_models/q8b_q30ba3b_ifrlvr_think_correct5917_sftinit_ifrlvr_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step516

SFT_NAME=q8b_mopd_math_code_instruct_full7168_exact_tokens_sft_b64_lr1e-5_ep1
SFT_CHECKPOINT_ROOT="checkpoint/${SFT_NAME}"
SFT_MODEL_PREFIX="merged_models/${SFT_NAME}"
MOPD_MODEL=merged_models/q8b_pragmatic_full7168sftinit_plain_mopd_8gpu_step484
EVAL_LAUNCHER=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/rjob_eval_q8b_mopd_plain_then_prefix128_8gpu.bash
MOPD_LAUNCHER=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/OPD/scripts/rjobs/archive/pragmatic/rjob_launch_q8b_pragmatic_full7168sftinit_plain_mopd_8gpu.bash

SFT_BATCH_SIZE=64
SFT_MICRO_BATCH_SIZE=1
SFT_MAX_LENGTH=9216
SFT_EPOCHS=1
SFT_LR=1e-5

check_hf_model() {
  local model=$1
  test -f "${model}/config.json" || { echo "Missing model config: ${model}/config.json" >&2; return 1; }
  compgen -G "${model}/*.safetensors" >/dev/null || {
    echo "Missing model weights: ${model}/*.safetensors" >&2
    return 1
  }
}

for required in \
  "${MATH_SOURCE}" "${CODE_SOURCE}" "${INSTRUCT_SOURCE}" \
  "${MATH_TEACHER}" "${CODE_TEACHER}" "${INSTRUCT_TEACHER}" \
  scripts/sft/make_teacher_prefix_sft_data.py \
  scripts/sft/precomputed_token_sft_dataset.py \
  "${EVAL_LAUNCHER}" "${MOPD_LAUNCHER}"; do
  test -e "${required}" || { echo "Missing required path: ${required}" >&2; exit 1; }
done
check_hf_model "${STUDENT_MODEL}"
check_hf_model "${MATH_TEACHER}"
check_hf_model "${CODE_TEACHER}"
check_hf_model "${INSTRUCT_TEACHER}"

# Parse both Hydra configurations without torchrun, Ray, CUDA, training, or evaluation.
if [[ "${PIPELINE_CONFIG_ONLY:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${SFT_DATASET}" data.val_files="${SFT_DATASET}" \
    data.multiturn.enable=False \
    data.custom_cls.path=scripts/sft/precomputed_token_sft_dataset.py \
    data.custom_cls.name=PrecomputedTokenSFTDataset \
    data.max_length="${SFT_MAX_LENGTH}" data.truncation=error \
    data.train_batch_size="${SFT_BATCH_SIZE}" data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE}" \
    +data.pad_mode=right model.partial_pretrain="${STUDENT_MODEL}" \
    model.trust_remote_code=True model.fsdp_config.model_dtype=bfloat16 \
    model.fsdp_config.offload_params=False model.enable_gradient_checkpointing=True \
    use_remove_padding=True optim.lr="${SFT_LR}" \
    trainer.default_local_dir="${SFT_CHECKPOINT_ROOT}" trainer.project_name=MOPD \
    trainer.experiment_name="${SFT_NAME}" trainer.total_epochs="${SFT_EPOCHS}" \
    trainer.seed=42 trainer.save_freq=1 trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard"]' trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes=1 trainer.resume_mode=auto trainer.checkpoint.save_contents='["hf_model"]' \
    trainer.checkpoint.load_contents='["hf_model"]' --cfg job >/dev/null
  SFT_MODEL_DIR="${STUDENT_MODEL}" MOPD_CONFIG_ONLY=1 bash "${MOPD_LAUNCHER}" >/dev/null
  echo "SFT and MOPD Hydra configuration parsing succeeded. No training or evaluation was started."
  exit 0
fi

echo "[1/6] Validate and assemble the 67,836 exact teacher trajectories"
"${PYTHON_BIN}" - \
  "${FULL_RESPONSE_SOURCE}" \
  "${MATH_SOURCE}" 17917 \
  "${CODE_SOURCE}" 25110 \
  "${INSTRUCT_SOURCE}" 24809 <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

output = Path(sys.argv[1])
specs = [tuple(sys.argv[i : i + 2]) for i in range(2, len(sys.argv), 2)]
expected_total = sum(int(rows) for _, rows in specs)

for path, expected_rows in specs:
    parquet_file = pq.ParquetFile(path)
    expected_rows = int(expected_rows)
    if parquet_file.metadata.num_rows != expected_rows:
        raise SystemExit(f"{path}: expected {expected_rows} rows, got {parquet_file.metadata.num_rows}")

if output.exists():
    existing = pq.ParquetFile(output)
    if existing.metadata.num_rows != expected_total:
        raise SystemExit(f"Existing {output} has {existing.metadata.num_rows} rows, expected {expected_total}")
    print(f"Reusing validated merged source: {output} ({expected_total} rows)")
    raise SystemExit(0)

output.parent.mkdir(parents=True, exist_ok=True)
temporary = Path(f"{output}.tmp")
if temporary.exists():
    temporary.unlink()
writer = None
try:
    for path, _ in specs:
        parquet_file = pq.ParquetFile(path)
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
finally:
    if writer is not None:
        writer.close()
temporary.replace(output)
print(f"Wrote globally ordered merged source: {output} ({expected_total} rows)")
PY

if [[ ! -f "${SFT_DATASET}" ]]; then
  echo "[2/6] Build exact-token full-response SFT data (max sequence length ${SFT_MAX_LENGTH})"
  "${PYTHON_BIN}" scripts/sft/make_teacher_prefix_sft_data.py \
    --input "${FULL_RESPONSE_SOURCE}" \
    --output "${SFT_DATASET}" \
    --tokenizer "${STUDENT_MODEL}" \
    --response-column teacher_response_text \
    --generated-token-ids-column teacher_response_token_ids \
    --finish-reason-column teacher_response_finish_reason \
    --max-length "${SFT_MAX_LENGTH}" \
    --enable-thinking \
    --use-generated-token-ids
else
  echo "[2/6] Reuse exact-token SFT data: ${SFT_DATASET}"
fi

SFT_ROWS=$("${PYTHON_BIN}" -c "import pyarrow.parquet as pq; print(pq.ParquetFile('${SFT_DATASET}').metadata.num_rows)")
SFT_EXPECTED_STEP=$((SFT_ROWS / SFT_BATCH_SIZE * SFT_EPOCHS))
test "${SFT_EXPECTED_STEP}" -gt 0 || { echo "SFT dataset is too small: ${SFT_ROWS}" >&2; exit 1; }
SFT_HF_DIR="${SFT_CHECKPOINT_ROOT}/global_step_${SFT_EXPECTED_STEP}/huggingface"
SFT_MODEL_DIR="${SFT_MODEL_PREFIX}_step${SFT_EXPECTED_STEP}"

echo "================================================================"
echo "student=${STUDENT_MODEL} | thinking=True"
echo "full SFT: rows=${SFT_ROWS}/67836 | max_length=${SFT_MAX_LENGTH} | global_batch=${SFT_BATCH_SIZE} | microbatch/GPU=${SFT_MICRO_BATCH_SIZE} | lr=${SFT_LR} | epochs=${SFT_EPOCHS}"
echo "full SFT final step=${SFT_EXPECTED_STEP} | save_freq=${SFT_EXPECTED_STEP} | model=${SFT_MODEL_DIR}"
echo "plain MOPD: response=4096 | model_len/token cap=6144 | batch=140 | lr=1e-6 | final step=484 | student/teacher GPUs=5/3"
echo "both checkpoints: Math Avg@16; HumanEval+/MBPP+ Avg@4; LCB n=10; IFEval n=1"
echo "================================================================"

if [[ ! -f "${SFT_MODEL_DIR}/config.json" ]] || ! compgen -G "${SFT_MODEL_DIR}/*.safetensors" >/dev/null; then
  test ! -e "${SFT_MODEL_DIR}" || { echo "Incomplete SFT export exists: ${SFT_MODEL_DIR}" >&2; exit 1; }
  echo "[3/6] Train three-domain full-response SFT"
  torchrun --standalone --nnodes=1 --nproc_per_node="${N_GPUS_PER_NODE}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${SFT_DATASET}" data.val_files="${SFT_DATASET}" \
    data.train_max_samples=-1 data.val_max_samples=256 \
    data.multiturn.enable=False \
    data.custom_cls.path=scripts/sft/precomputed_token_sft_dataset.py \
    data.custom_cls.name=PrecomputedTokenSFTDataset \
    data.max_length="${SFT_MAX_LENGTH}" data.truncation=error \
    data.train_batch_size="${SFT_BATCH_SIZE}" data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE}" \
    +data.pad_mode=right \
    model.partial_pretrain="${STUDENT_MODEL}" model.trust_remote_code=True \
    model.fsdp_config.model_dtype=bfloat16 model.fsdp_config.offload_params=False \
    model.enable_gradient_checkpointing=True use_remove_padding=True \
    optim.lr="${SFT_LR}" \
    trainer.default_local_dir="${SFT_CHECKPOINT_ROOT}" trainer.project_name=MOPD \
    trainer.experiment_name="${SFT_NAME}" trainer.total_epochs="${SFT_EPOCHS}" \
    trainer.seed=42 trainer.save_freq="${SFT_EXPECTED_STEP}" trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard"]' trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" trainer.nnodes=1 \
    trainer.max_ckpt_to_keep=1 trainer.resume_mode=auto \
    trainer.checkpoint.save_contents='["hf_model"]' trainer.checkpoint.load_contents='["hf_model"]'

  check_hf_model "${SFT_HF_DIR}"
  SFT_STAGE="${SFT_MODEL_DIR}.tmp.$$"
  cp -a "${SFT_HF_DIR}" "${SFT_STAGE}"
  check_hf_model "${SFT_STAGE}"
  mv "${SFT_STAGE}" "${SFT_MODEL_DIR}"
else
  echo "[3/6] Reuse complete SFT model: ${SFT_MODEL_DIR}"
fi
check_hf_model "${SFT_MODEL_DIR}"

echo "[4/6] Evaluate the SFT checkpoint on Math, Code, and IFEval"
EVAL_TARGET=custom \
CUSTOM_MODEL="${SFT_MODEL_DIR}" \
CUSTOM_LABEL=q8b_mopd_full7168_sft \
bash "${EVAL_LAUNCHER}"

echo "[5/6] Train, checkpoint, and merge plain MOPD initialized from the SFT model"
SFT_MODEL_DIR="${SFT_MODEL_DIR}" bash "${MOPD_LAUNCHER}"
check_hf_model "${MOPD_MODEL}"

echo "[6/6] Evaluate the SFT+plain-MOPD checkpoint on Math, Code, and IFEval"
EVAL_TARGET=custom \
CUSTOM_MODEL="${MOPD_MODEL}" \
CUSTOM_LABEL=q8b_mopd_full7168_sftinit_plain_mopd \
bash "${EVAL_LAUNCHER}"

echo "Completed full SFT and SFT+plain-MOPD training, checkpoint publication, and both final evaluations."
