#!/usr/bin/env bash
# Hint-assisted OPD ablation:
#   Qwen3-1.7B-Base student <- Qwen3-4B-Base-GRPO teacher (math)
#
# The student rollout sees x + <HINT>...</HINT>.  Immediately after rollout,
# the trainer rebuilds x + y and computes both teacher and student log-probs
# without the hint.  This is OPD, with no verifier filtering or true-reward
# term in the loss.  Base models use their official thinking-enabled template.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl

# rjob starts this launcher outside the OPD repository, so a relative
# `source .env` is not reliable.
source /mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/OPD/.env
cd "${OPD_ROOT}"
export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

INPUT_DATASET="${HINT_INPUT_DATASET:-datasets/dapo-math-17k-teacher-aligned.parquet}"
HINT_DATASET="${HINT_DATASET:-datasets/teacher_hint/q3b_q4bgrpo_dapo_math17k_hint_tagged.parquet}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen3-4B-Base-GRPO}"
STUDENT_MODEL="${STUDENT_MODEL:-${MODEL_ROOT}/Qwen3-1.7B-Base}"

test -f "${INPUT_DATASET}"
test -f "${TEACHER_MODEL}/config.json"
test -f "${STUDENT_MODEL}/config.json"

HINT_LIMIT_ARGS=()
if [[ -n "${HINT_LIMIT:-}" ]]; then
    HINT_LIMIT_ARGS+=(--limit "${HINT_LIMIT}")
fi

HINT_THINKING_ARGS=()
case "${HINT_ENABLE_THINKING:-True}" in
    True|true) HINT_THINKING_ARGS+=(--enable-thinking) ;;
    False|false) ;;
    *)
        echo "HINT_ENABLE_THINKING must be True or False; got ${HINT_ENABLE_THINKING}" >&2
        exit 1
        ;;
esac

if [ ! -f "${HINT_DATASET}" ]; then
    python scripts/teacher_hint/generate_teacher_hint_data.py \
        --input "${INPUT_DATASET}" \
        --output "${HINT_DATASET}" \
        --teacher-model "${TEACHER_MODEL}" \
        --gpus "${HINT_GPUS:-0,1,2,3}" \
        --tensor-parallel-size "${HINT_TENSOR_PARALLEL_SIZE:-4}" \
        --max-model-len "${HINT_MAX_MODEL_LEN:-4096}" \
        --max-new-tokens "${HINT_MAX_NEW_TOKENS:-2048}" \
        --temperature 0.7 \
        --top-p 0.95 \
        --batch-size "${HINT_BATCH_SIZE:-256}" \
        "${HINT_THINKING_ARGS[@]}" \
        "${HINT_LIMIT_ARGS[@]}" \
        --force
else
    echo "Reusing hint dataset: ${HINT_DATASET}"
fi

export RUN_MODE=plain
export ENABLE_THINKING=True
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
export DATA_SHUFFLE=True
export DATA_SEED=42
export ACTOR_MODEL_PATH="${STUDENT_MODEL}"
export REWARD_MODEL_PATH="${TEACHER_MODEL}"
export PLAIN_TRAIN_DATASET="${HINT_DATASET}"
export PLAIN_TRAIN_DATASET_NAME="q3b_q4bgrpo_hint_assisted_offline_opd"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q3b_q4bgrpo_hint_assisted_offline_opd_r4096_b64_n1_lr1e-5_$(date +%Y-%m-%d_%H-%M-%S)}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"
# The generic plain-OPD script derives its final model name from this variable.
# It must be run-specific; otherwise repeated smoke runs all reuse step4 evals.
export PLAIN_MODEL_OUTPUT_NAME_PREFIX="${EXPERIMENT_NAME}"
export TEACHER_PREFIX_MAX_LEN=0
export TEACHER_PREFIX_SFT_LOSS_COEF=0.0
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0
export MAX_PROMPT_LENGTH=2048
export MAX_RESP_LENGTH=4096
export MAX_VAL_RESP_LENGTH=4096
export TOTAL_EPOCHS=1
export LR="${LR:-1e-5}"
export EVAL_MAX_TOKENS=7168
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OPD_ROOT}/outputs/eval/hint_assisted_offline_opd}"

# This is the only experiment-specific switch. It changes the scoring/update
# prompt after hint-conditioned rollout; rollout itself remains hint-aware.
export EXTRA_PPO_ARGS="+data.remove_teacher_hint_for_training=True"

bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
