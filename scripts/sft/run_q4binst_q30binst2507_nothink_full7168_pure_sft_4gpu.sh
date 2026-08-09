#!/usr/bin/env bash
# Full-response pure SFT control:
# Qwen3-4B-Instruct-2507 <- Qwen3-30B-A3B-Instruct-2507, native no-think.
# Supervise the exact saved teacher token IDs to natural EOS or the original
# 7168-token teacher rollout cap, then merge and run the 9192-token math eval.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200

FULL_RESPONSE_DATASET="${FULL_RESPONSE_DATASET:-datasets/sft_teacher_response/q4binst_q30binst2507_nothink_full_response_7168.parquet}"
SFT_DATASET="${SFT_DATASET:-datasets/sft/q4binst_q30binst2507_nothink_full7168_pure_sft.parquet}"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_full7168_pure_sft_b96_lr1e-5}"
SFT_MODEL_NAME="${SFT_MODEL_NAME:-q4binst_q30binst2507_nothink_full7168_pure_sft_b96_lr1e-5}"
SFT_RUN_TAG="${SFT_RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export N_GPUS_PER_NODE=4
export EVAL_GPUS=0,1,2,3
export LR="${LR:-1e-5}"
export TOTAL_EPOCHS=1
export TRAIN_BATCH_SIZE=96
export MICRO_BATCH_SIZE_PER_GPU=1
export DATA_SEED=42
export ENABLE_THINKING=False
export EVAL_MAX_TOKENS=9192
export EVAL_OUTPUT_DIR="${OPD_ROOT}/outputs/eval/q4binst_q30binst2507_nothink_full7168_pure_sft"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${SFT_EXPERIMENT_NAME}_${SFT_RUN_TAG}}"

test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${FULL_RESPONSE_DATASET}" || {
    echo "Missing saved 7k teacher trajectories: ${FULL_RESPONSE_DATASET}" >&2
    exit 1
}

# This is the complete SFT sequence budget, not the teacher target cap: the
# latter remains 7168 because it comes from the saved full-response asset.
MAX_LENGTH=9216
python scripts/sft/make_teacher_prefix_sft_data.py \
    --input "${FULL_RESPONSE_DATASET}" \
    --output "${SFT_DATASET}" \
    --tokenizer "${ACTOR_MODEL_PATH}" \
    --response-column teacher_response_text \
    --generated-token-ids-column teacher_response_token_ids \
    --finish-reason-column teacher_response_finish_reason \
    --max-length "${MAX_LENGTH}" \
    --use-generated-token-ids

EXPECTED_STEPS=$(/root/miniconda3/envs/verl/bin/python -c "import pyarrow.parquet as pq; n=pq.ParquetFile('${SFT_DATASET}').metadata.num_rows; print(max(1, n // ${TRAIN_BATCH_SIZE}))")
SFT_ROWS=$(/root/miniconda3/envs/verl/bin/python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('${SFT_DATASET}').metadata.num_rows)")
echo "Full-response SFT rows / expected steps: ${SFT_ROWS} / ${EXPECTED_STEPS}"

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
    trainer.experiment_name="${SFT_EXPERIMENT_NAME}" \
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
if [ "${STEP}" -lt "${EXPECTED_STEPS}" ]; then
    echo "SFT stopped at step ${STEP}; expected ${EXPECTED_STEPS}. Skip merge/eval." >&2
    exit 1
fi

HF_DIR="${CKPT_PATH}/global_step_${STEP}/huggingface"
MODEL_DIR="${OPD_ROOT}/merged_models/${SFT_MODEL_NAME}_step${STEP}"
test -f "${HF_DIR}/config.json" || { echo "Missing SFT HuggingFace checkpoint: ${HF_DIR}" >&2; exit 1; }
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    mkdir -p "$(dirname "${MODEL_DIR}")"
    cp -a "${HF_DIR}" "${MODEL_DIR}"
fi

if [ "${SKIP_FINAL_EVAL:-False}" = "True" ]; then
    echo "SKIP_FINAL_EVAL=True: merged SFT checkpoint is ready; skipping the math-only final evaluator."
    echo "SFT_MODEL_DIR=${MODEL_DIR}"
    exit 0
fi

EVAL_DIR="${EVAL_OUTPUT_DIR}/$(basename "${MODEL_DIR}")"
python scripts/val/eval/gen_vllm.py \
    --model "${MODEL_DIR}" \
    --data-dir "${OPD_ROOT}/scripts/val/data" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --tasks AIME24,AIME25,AMC23 \
    --n 16 \
    --max-tokens "${EVAL_MAX_TOKENS}" \
    --temperature 0.7 \
    --top-p 0.95 \
    --gpus "${EVAL_GPUS}" \
    --disable-thinking
python scripts/val/eval/grade.py --eval-dir "${EVAL_DIR}"

echo "SFT_MODEL_DIR=${MODEL_DIR}"
