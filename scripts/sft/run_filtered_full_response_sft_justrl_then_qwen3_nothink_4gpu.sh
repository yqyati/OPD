#!/bin/bash
#SBATCH --job-name=filtered-full-sft
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
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}
export LR=${LR:-1e-5}
export MAX_LENGTH=${MAX_LENGTH:-8192}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-/tmp/opd_logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/filtered_full_sft_sequence_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

run_filtered_full_sft_stage() {
    local stage_name="$1"
    local student_model="$2"
    local response_data="$3"
    local filtered_response_data="$4"
    local sft_data="$5"
    local experiment_name="$6"
    local model_name="$7"

    if [ ! -f "$response_data" ]; then
        echo "Missing full teacher response data: $response_data" >&2
        exit 1
    fi

    echo "========== Filter correct teacher responses: ${stage_name} =========="
    python scripts/sft/filter_teacher_response_correct.py \
        --input "$response_data" \
        --output "$filtered_response_data" \
        --response-column teacher_response_text

    echo "========== Run filtered full-response SFT: ${stage_name} =========="
    ACTOR_MODEL_PATH="$student_model" \
    SOURCE_PREFIX_DATA="$filtered_response_data" \
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

run_filtered_full_sft_stage \
    "justrl_filtered_full_response_sft" \
    "/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/DeepSeek-R1-Distill-Qwen-1.5B" \
    "datasets/sft_teacher_response/justrl_full_response_7168.parquet" \
    "datasets/sft_teacher_response/justrl_full_response_7168_correct.parquet" \
    "datasets/sft/justrl_full_response_sft_7168_correct.parquet" \
    "justrl_filtered_full_response_sft_lr${LR}" \
    "justrl_filtered_full_response_sft_lr${LR}"

run_filtered_full_sft_stage \
    "qwen3_nothink_filtered_full_response_sft" \
    "/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/Qwen3-1.7B-Base" \
    "datasets/sft_teacher_response/qwen3_base_nothink_full_response_7168.parquet" \
    "datasets/sft_teacher_response/qwen3_base_nothink_full_response_7168_correct.parquet" \
    "datasets/sft/qwen3_nothink_full_response_sft_7168_correct.parquet" \
    "qwen3_nothink_filtered_full_response_sft_lr${LR}" \
    "qwen3_nothink_filtered_full_response_sft_lr${LR}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
