#!/bin/bash
#SBATCH --job-name=full-sft
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

set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}
export LR=${LR:-1e-5}
export MAX_LENGTH=${MAX_LENGTH:-8192}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-7168}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
export INPUT_DATASET=${INPUT_DATASET:-datasets/dapo-math-17k-teacher-aligned.parquet}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-/tmp/opd_logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/full_sft_sequence_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

run_full_sft_stage() {
    local stage_name="$1"
    local teacher_model="$2"
    local student_model="$3"
    local response_data="$4"
    local sft_data="$5"
    local experiment_name="$6"
    local model_name="$7"

    echo "========== Generate full teacher responses: ${stage_name} =========="
    python scripts/sft/generate_teacher_response_data.py \
        --input "$INPUT_DATASET" \
        --output "$response_data" \
        --teacher-model "$teacher_model" \
        --gpus "$EVAL_GPUS" \
        --tensor-parallel-size "$N_GPUS_PER_NODE" \
        --max-model-len "$MAX_LENGTH" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature 0.7 \
        --top-p 0.95 \
        --batch-size 64

    echo "========== Run full-response SFT: ${stage_name} =========="
    ACTOR_MODEL_PATH="$student_model" \
    SOURCE_PREFIX_DATA="$response_data" \
    SFT_DATASET="$sft_data" \
    RESPONSE_COLUMN=teacher_response_text \
    EXPERIMENT_NAME="$experiment_name" \
    MODEL_NAME="$model_name" \
    LR="$LR" \
    MAX_LENGTH="$MAX_LENGTH" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    MICRO_BATCH_SIZE_PER_GPU="$MICRO_BATCH_SIZE_PER_GPU" \
    N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
    EVAL_GPUS="$EVAL_GPUS" \
    bash scripts/sft/run_qwen3_grpo_teacher_prefix128_pure_sft_4gpu.sh
}

run_full_sft_stage \
    "justrl_full_response_sft" \
    "/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/JustRL-DeepSeek-1.5B" \
    "/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/DeepSeek-R1-Distill-Qwen-1.5B" \
    "datasets/sft_teacher_response/justrl_full_response_7168.parquet" \
    "datasets/sft/justrl_full_response_sft_7168.parquet" \
    "justrl_full_response_sft_lr${LR}" \
    "justrl_full_response_sft_lr${LR}"

run_full_sft_stage \
    "qwen3_nothink_full_response_sft" \
    "/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/Qwen3-4B-Base" \
    "/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/Qwen3-1.7B-Base" \
    "datasets/sft_teacher_response/qwen3_base_nothink_full_response_7168.parquet" \
    "datasets/sft/qwen3_nothink_full_response_sft_7168.parquet" \
    "qwen3_nothink_full_response_sft_lr${LR}" \
    "qwen3_nothink_full_response_sft_lr${LR}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
