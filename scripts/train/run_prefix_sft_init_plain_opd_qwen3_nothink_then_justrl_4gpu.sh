#!/bin/bash
#SBATCH --job-name=prefix-sft-opd
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

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}
export LR=${LR:-1e-5}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-/tmp/opd_logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/prefix_sft_init_plain_opd_sequence_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

echo "========== Stage 1: Qwen3 no-think prefix-SFT init -> plain OPD =========="
N_GPUS_PER_NODE="$N_GPUS_PER_NODE" EVAL_GPUS="$EVAL_GPUS" LR="$LR" \
    bash scripts/train/run_qwen3_nothink_prefix_sft_init_plain_opd_4gpu.sh

echo "========== Stage 2: JustRL prefix-SFT init -> plain OPD =========="
N_GPUS_PER_NODE="$N_GPUS_PER_NODE" EVAL_GPUS="$EVAL_GPUS" LR="$LR" \
    bash scripts/train/run_justrl_prefix_sft_init_plain_opd_4gpu.sh

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
