#!/usr/bin/env bash
# Pure SFT warm start on the exact, completion-aware 512-token trajectories:
# Qwen3-4B-Instruct-2507 <- Qwen3-30B-A3B-Instruct-2507, native no-think.
#
# This script intentionally stops after merging the SFT checkpoint.  The
# following plain-OPD stage is owned by the top-level sequence launcher.
set -euo pipefail
set -x

source .env
cd "${OPD_ROOT}"

export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200

export N_GPUS_PER_NODE=4
export LR="${LR:-1e-5}"
export TOTAL_EPOCHS=1
export MAX_LENGTH=2048
export TRAIN_BATCH_SIZE=96
export MICRO_BATCH_SIZE_PER_GPU=1
export DATA_SEED=42
export ENABLE_THINKING=False

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export SOURCE_PREFIX_DATA="${SOURCE_PREFIX_DATA:-datasets/teacher_prefix/q4binst_q30binst2507_nothink_dapo_math17k_prefix512.parquet}"
export SFT_DATASET="${SFT_DATASET:-datasets/sft/q4binst_q30binst2507_nothink_prefix512_pure_sft.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_prefix512_pure_sft_b96_lr1e-5}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"
export MODEL_NAME="${MODEL_NAME:-q4binst_q30binst2507_nothink_prefix512_pure_sft_b96_lr1e-5}"

test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${SOURCE_PREFIX_DATA}" || { echo "Missing prefix data: ${SOURCE_PREFIX_DATA}" >&2; exit 1; }

# The prefix-text field is deliberately empty for this dataset.  Consume exact
# generated token IDs so SFT sees precisely the teacher trajectory that the
# fixed-prefix OPD experiment used (including a natural im_end when it stopped).
python scripts/sft/make_teacher_prefix_sft_data.py \
    --input "${SOURCE_PREFIX_DATA}" \
    --output "${SFT_DATASET}" \
    --tokenizer "${ACTOR_MODEL_PATH}" \
    --response-column teacher_prefix_text \
    --generated-token-ids-column teacher_prefix_token_ids \
    --finish-reason-column teacher_prefix_finish_reason \
    --max-length "${MAX_LENGTH}" \
    --use-generated-token-ids

EXPECTED_STEPS=$(/root/miniconda3/envs/verl/bin/python -c "import pyarrow.parquet as pq; n=pq.ParquetFile('${SFT_DATASET}').metadata.num_rows; print(max(1, n // ${TRAIN_BATCH_SIZE}))")
SFT_ROWS=$(/root/miniconda3/envs/verl/bin/python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('${SFT_DATASET}').metadata.num_rows)")
echo "SFT rows / expected steps: ${SFT_ROWS} / ${EXPECTED_STEPS}"

torchrun --standalone --nnodes=1 --nproc_per_node="${N_GPUS_PER_NODE}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${SFT_DATASET}" \
    data.val_files="${SFT_DATASET}" \
    data.train_max_samples=-1 \
    data.val_max_samples=256 \
    data.multiturn.enable=False \
    data.custom_cls.path=scripts/sft/precomputed_token_sft_dataset.py \
    data.custom_cls.name=PrecomputedTokenSFTDataset \
    data.max_length="${MAX_LENGTH}" \
    data.truncation=error \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    +data.pad_mode=right \
    model.partial_pretrain="${ACTOR_MODEL_PATH}" \
    model.trust_remote_code=True \
    model.fsdp_config.model_dtype=bfloat16 \
    model.fsdp_config.offload_params=False \
    model.enable_gradient_checkpointing=True \
    use_remove_padding=True \
    optim.lr="${LR}" \
    trainer.default_local_dir="${CKPT_PATH}" \
    trainer.project_name=OnPolicyDistillation \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.seed="${DATA_SEED}" \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard"]' \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.max_ckpt_to_keep=1 \
    trainer.resume_mode=disable \
    trainer.checkpoint.save_contents='["hf_model"]' \
    trainer.checkpoint.load_contents='["hf_model"]'

STEP="${STEP:-latest}"
if [ "${STEP}" = latest ]; then
    STEP=$(find "${CKPT_PATH}" -maxdepth 1 -type d -name 'global_step_*' \
        | sed -E 's#.*/global_step_([0-9]+)$#\1#' | sort -n | tail -1)
fi
test -n "${STEP}" || { echo "No SFT checkpoint found under ${CKPT_PATH}" >&2; exit 1; }

HF_DIR="${CKPT_PATH}/global_step_${STEP}/huggingface"
MODEL_DIR="${OPD_ROOT}/merged_models/${MODEL_NAME}_step${STEP}"
test -f "${HF_DIR}/config.json" || { echo "Missing SFT HuggingFace checkpoint: ${HF_DIR}" >&2; exit 1; }
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    mkdir -p "$(dirname "${MODEL_DIR}")"
    cp -a "${HF_DIR}" "${MODEL_DIR}"
fi

echo "SFT_MODEL_DIR=${MODEL_DIR}"
