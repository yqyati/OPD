#!/bin/bash
#SBATCH --job-name=qwen3-sft
#SBATCH --output=logs/20251004/output_%j.log
#SBATCH --error=logs/20251004/error_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

source .env

set -euo pipefail
set -x

cd ${OPD_ROOT}

export PYTHONPATH=${OPD_ROOT}/verl:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-/tmp/opd_logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/pure_sft_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}
export LR=${LR:-1e-5}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export MAX_LENGTH=${MAX_LENGTH:-2048}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
export DATA_SEED=${DATA_SEED:-42}
export ENABLE_THINKING=${ENABLE_THINKING:-False}
export USE_GENERATED_TOKEN_IDS=${USE_GENERATED_TOKEN_IDS:-False}
export GENERATED_TOKEN_IDS_COLUMN=${GENERATED_TOKEN_IDS_COLUMN:-teacher_prefix_token_ids}
export GENERATED_FINISH_REASON_COLUMN=${GENERATED_FINISH_REASON_COLUMN:-teacher_prefix_finish_reason}
export STUDENT_CHAT_TEMPLATE_FILE=${STUDENT_CHAT_TEMPLATE_FILE:-}
export SOURCE_EOS_TOKEN_ID=${SOURCE_EOS_TOKEN_ID:-}
export CANONICAL_EOS_TOKEN_ID=${CANONICAL_EOS_TOKEN_ID:-}
export EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-15000}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-1.7B-Base}
export SOURCE_PREFIX_DATA=${SOURCE_PREFIX_DATA:-datasets/teacher_prefix/qwen3_grpo_dapo_math_17k_teacher_prefix128.parquet}
export SFT_DATASET=${SFT_DATASET:-datasets/sft/qwen3_grpo_teacher_prefix128_pure_sft.parquet}
export RESPONSE_COLUMN=${RESPONSE_COLUMN:-teacher_prefix_text}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_grpo_teacher_prefix128_pure_sft_lr${LR}}
export CKPT_PATH=${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}_$(date +%Y-%m-%d_%H-%M-%S)}
export MODEL_NAME=${MODEL_NAME:-qwen3_grpo_teacher_prefix128_pure_sft_lr${LR}}
export OUTPUT_DIR=${OUTPUT_DIR:-${OPD_ROOT}/outputs/eval/justrl_eval_outputs_15000}
export DATA_DIR=${OPD_ROOT}/scripts/val/data

case "${ENABLE_THINKING,,}" in
    true) THINKING_ARG=(--enable-thinking); EVAL_THINKING_ARG=--enable-thinking ;;
    false) THINKING_ARG=(); EVAL_THINKING_ARG=--disable-thinking ;;
    *) echo "ENABLE_THINKING must be True or False" >&2; exit 1 ;;
esac
case "${USE_GENERATED_TOKEN_IDS,,}" in
    true)
        TOKEN_ID_ARG=(--use-generated-token-ids)
        CUSTOM_DATASET_ARGS=(
            data.multiturn.enable=False
            data.custom_cls.path=scripts/sft/precomputed_token_sft_dataset.py
            data.custom_cls.name=PrecomputedTokenSFTDataset
        )
        ;;
    false) TOKEN_ID_ARG=(); CUSTOM_DATASET_ARGS=(data.multiturn.enable=True) ;;
    *) echo "USE_GENERATED_TOKEN_IDS must be True or False" >&2; exit 1 ;;
esac

SFT_TEMPLATE_ARGS=()
EVAL_TEMPLATE_ARGS=()
if [ -n "${STUDENT_CHAT_TEMPLATE_FILE}" ]; then
    if [ ! -f "${STUDENT_CHAT_TEMPLATE_FILE}" ]; then
        echo "Missing student chat template: ${STUDENT_CHAT_TEMPLATE_FILE}" >&2
        exit 1
    fi
    SFT_TEMPLATE_ARGS+=(--chat-template-file "${STUDENT_CHAT_TEMPLATE_FILE}")
    EVAL_TEMPLATE_ARGS+=(--prompt-template-file "${STUDENT_CHAT_TEMPLATE_FILE}")
fi

SFT_EOS_ARGS=()
EVAL_EOS_ARGS=()
if [ -n "${SOURCE_EOS_TOKEN_ID}" ] || [ -n "${CANONICAL_EOS_TOKEN_ID}" ]; then
    if [ -z "${SOURCE_EOS_TOKEN_ID}" ] || [ -z "${CANONICAL_EOS_TOKEN_ID}" ]; then
        echo "SOURCE_EOS_TOKEN_ID and CANONICAL_EOS_TOKEN_ID must be set together." >&2
        exit 1
    fi
    SFT_EOS_ARGS+=(
        --source-eos-token-id "${SOURCE_EOS_TOKEN_ID}"
        --canonical-eos-token-id "${CANONICAL_EOS_TOKEN_ID}"
    )
    EVAL_EOS_ARGS+=(--stop-token-ids "${CANONICAL_EOS_TOKEN_ID}")
fi

if [ ! -f "$SOURCE_PREFIX_DATA" ]; then
    echo "Missing source teacher-prefix data: $SOURCE_PREFIX_DATA" >&2
    exit 1
fi

python scripts/sft/make_teacher_prefix_sft_data.py \
    --input "$SOURCE_PREFIX_DATA" \
    --output "$SFT_DATASET" \
    --tokenizer "$ACTOR_MODEL_PATH" \
    --response-column "$RESPONSE_COLUMN" \
    --generated-token-ids-column "$GENERATED_TOKEN_IDS_COLUMN" \
    --finish-reason-column "$GENERATED_FINISH_REASON_COLUMN" \
    --max-length "$MAX_LENGTH" \
    "${THINKING_ARG[@]}" \
    "${TOKEN_ID_ARG[@]}" \
    "${SFT_TEMPLATE_ARGS[@]}" \
    "${SFT_EOS_ARGS[@]}"

EXPECTED_STEPS=$(/root/miniconda3/envs/verl/bin/python -c "import pandas as pd; n=len(pd.read_parquet('${SFT_DATASET}')); bs=${TRAIN_BATCH_SIZE}; ep=${TOTAL_EPOCHS}; print(max(1, (n // bs) * ep))")
echo "EXPECTED_STEPS: ${EXPECTED_STEPS}"
echo "CKPT_PATH: ${CKPT_PATH}"

torchrun --standalone --nnodes=1 --nproc_per_node="$N_GPUS_PER_NODE" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$SFT_DATASET" \
    data.val_files="$SFT_DATASET" \
    data.train_max_samples=-1 \
    data.val_max_samples=256 \
    "${CUSTOM_DATASET_ARGS[@]}" \
    data.multiturn.messages_key=messages \
    data.multiturn.enable_thinking_key=enable_thinking \
    data.max_length="$MAX_LENGTH" \
    data.truncation=error \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
    +data.pad_mode=right \
    model.partial_pretrain="$ACTOR_MODEL_PATH" \
    model.trust_remote_code=True \
    model.fsdp_config.model_dtype=bfloat16 \
    model.fsdp_config.offload_params=False \
    model.enable_gradient_checkpointing=True \
    use_remove_padding=True \
    optim.lr="$LR" \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.project_name=OnPolicyDistillation \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.seed="$DATA_SEED" \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard"]' \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.max_ckpt_to_keep=1 \
    trainer.resume_mode=disable \
    trainer.checkpoint.save_contents='["hf_model"]' \
    trainer.checkpoint.load_contents='["hf_model"]'

STEP=${STEP:-latest}
if [ "$STEP" = "latest" ]; then
    STEP=$(find "$CKPT_PATH" -maxdepth 1 -type d -name 'global_step_*' \
        | sed -E 's#.*/global_step_([0-9]+)$#\1#' \
        | sort -n \
        | tail -1)
fi

if [ -z "$STEP" ]; then
    echo "No saved checkpoint found under ${CKPT_PATH}" >&2
    exit 1
fi

HF_DIR="${CKPT_PATH}/global_step_${STEP}/huggingface"
MODEL_DIR="${OPD_ROOT}/merged_models/${MODEL_NAME}_step${STEP}"
EVAL_DIR="${OUTPUT_DIR}/$(basename "$MODEL_DIR")"

if [ ! -f "${HF_DIR}/config.json" ]; then
    echo "Missing HF checkpoint: ${HF_DIR}/config.json" >&2
    exit 1
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    mkdir -p "$(dirname "$MODEL_DIR")"
    cp -a "$HF_DIR" "$MODEL_DIR"
fi

python scripts/val/eval/gen_vllm.py \
    --model "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --tasks AIME24,AIME25,AMC23 \
    --n 16 \
    --max-tokens "$EVAL_MAX_TOKENS" \
    --temperature 0.7 \
    --top-p 0.95 \
    --gpus "$EVAL_GPUS" \
    "${EVAL_TEMPLATE_ARGS[@]}" \
    "${EVAL_EOS_ARGS[@]}" \
    "$EVAL_THINKING_ARG"

python scripts/val/eval/grade.py \
    --eval-dir "$EVAL_DIR"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
