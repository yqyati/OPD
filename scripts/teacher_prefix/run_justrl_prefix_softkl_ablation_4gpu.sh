#!/bin/bash
set -euo pipefail
set -x

cd /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD

export PYTHONPATH=/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD/verl:${PYTHONPATH:-}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}
export LR=${LR:-1e-5}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export TEACHER_PREFIX_SFT_LOSS_COEF=${TEACHER_PREFIX_SFT_LOSS_COEF:-0.1}
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=${TEACHER_PREFIX_SOFT_KL_LOSS_COEF:-0.01}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}

SOURCE_PREFIX_DATA=${SOURCE_PREFIX_DATA:-datasets/teacher_prefix/opd_prompt_all_teacher_prefix128.parquet}
SOFTKL_PREFIX_DATA=${SOFTKL_PREFIX_DATA:-datasets/teacher_prefix/opd_prompt_all_teacher_prefix128_topk${LOG_PROB_TOP_K}.parquet}
STUDENT_MODEL=${STUDENT_MODEL:-/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/JustRL-DeepSeek-1.5B}

if [ ! -f "$SOURCE_PREFIX_DATA" ]; then
    echo "Missing source teacher-prefix data: $SOURCE_PREFIX_DATA" >&2
    exit 1
fi

if [ ! -f "$SOFTKL_PREFIX_DATA" ]; then
    python scripts/teacher_prefix/generate_teacher_prefix_topk.py \
        --input "$SOURCE_PREFIX_DATA" \
        --output "$SOFTKL_PREFIX_DATA" \
        --teacher-model "$TEACHER_MODEL" \
        --tokenizer "$STUDENT_MODEL" \
        --top-k "$LOG_PROB_TOP_K" \
        --batch-size "${PREFIX_TOPK_BATCH_SIZE:-4}" \
        --max-length "$MAX_PROMPT_LENGTH"
else
    echo "Reuse existing prefix soft-KL data: $SOFTKL_PREFIX_DATA"
fi

RUN_MODE=prefix \
TRAIN_DATASET="$SOFTKL_PREFIX_DATA" \
TRAIN_DATASET_NAME="TP128-SFT0.1-SoftKL0.01-SuffixOPD" \
MODEL_OUTPUT_NAME_PREFIX="tp128_sft0.1_softkl0.01_suffixopd_lr${LR}" \
ACTOR_MODEL_PATH="$STUDENT_MODEL" \
ACTOR_MODEL_NAME="DSR1Qwen1.5B" \
REWARD_MODEL_PATH="$TEACHER_MODEL" \
REWARD_MODEL_NAME="JustRL1.5B" \
TEACHER_PREFIX_SFT_LOSS_COEF="$TEACHER_PREFIX_SFT_LOSS_COEF" \
TEACHER_PREFIX_SOFT_KL_LOSS_COEF="$TEACHER_PREFIX_SOFT_KL_LOSS_COEF" \
LR="$LR" \
MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
LOG_PROB_TOP_K="$LOG_PROB_TOP_K" \
N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
EVAL_GPUS="$EVAL_GPUS" \
bash scripts/train/run_justrl_prefix_sft_init_plain_opd_4gpu.sh
