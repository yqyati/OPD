#!/bin/bash
set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:${PYTHONPATH:-}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}
export LR=${LR:-1e-5}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-9000}
export TEACHER_PREFIX_SFT_LOSS_COEF=${TEACHER_PREFIX_SFT_LOSS_COEF:-0.1}
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}

SOURCE_SFT_DATA=${SOURCE_SFT_DATA:-datasets/sft_teacher_response/qwen3_base_nothink_full_response_7168.parquet}
MIXED_DATA=${MIXED_DATA:-datasets/mixed_sampling/qwen3_nothink_full_response_mixed_sft_plain_opd.parquet}
STUDENT_MODEL=${STUDENT_MODEL:-/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/Qwen3-1.7B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/Qwen3-4B-Base}

python scripts/mixed_sampling/build_mixed_prefix_sft_opd_dataset.py \
    --input "$SOURCE_SFT_DATA" \
    --output "$MIXED_DATA" \
    --target-column teacher_response_text \
    --teacher-every 2

RUN_MODE=prefix \
TRAIN_DATASET="$MIXED_DATA" \
TRAIN_DATASET_NAME="MixedFullSFT0.1-PlainOPD-Qwen3NoThink" \
MODEL_OUTPUT_NAME_PREFIX="mixed_full_sft0.1_plain_opd_qwen3_nothink_lr${LR}" \
ACTOR_MODEL_PATH="$STUDENT_MODEL" \
ACTOR_MODEL_NAME="Qwen3-1.7B-Base" \
REWARD_MODEL_PATH="$TEACHER_MODEL" \
REWARD_MODEL_NAME="Qwen3-4B-Base" \
TEACHER_PREFIX_SFT_LOSS_COEF="$TEACHER_PREFIX_SFT_LOSS_COEF" \
TEACHER_PREFIX_SOFT_KL_LOSS_COEF="$TEACHER_PREFIX_SOFT_KL_LOSS_COEF" \
LR="$LR" \
MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
LOG_PROB_TOP_K="$LOG_PROB_TOP_K" \
N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
EVAL_GPUS="$EVAL_GPUS" \
bash scripts/train/run_qwen3_nothink_prefix_sft_init_plain_opd_4gpu.sh
