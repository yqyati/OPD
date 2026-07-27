#!/usr/bin/env bash
# Fixed-prefix-128 OPD: Qwen3-4B-Instruct-2507 <- Qwen3-30B-A3B-Instruct-2507.
# The script defines local training + merge + math evaluation only; it never
# submits an rjob.  Generate the matching prefix parquet first with:
#   bash scripts/teacher_prefix/run_q4binst_q30binst2507_nothink_prefix128_4gpu.sh
set -euo pipefail

source .env
cd "${OPD_ROOT}"

# Keep plain-OPD's model pair, native Instruct prompt contract, response cap,
# shuffle seed, batch size, LR, and evaluator.  The sole method change is the
# fixed 128-token teacher prefix plus its prefix-token CE coefficient.
export RUN_MODE=prefix
export PREFIX_TRAIN_DATASET="${PREFIX_TRAIN_DATASET:-datasets/teacher_prefix/q4binst_q30binst2507_nothink_dapo_math17k_prefix128.parquet}"
export PREFIX_TRAIN_DATASET_NAME="q4binst_q30binst2507_nothink_prefix128_completionaware"
export TEACHER_PREFIX_SFT_LOSS_COEF="${TEACHER_PREFIX_SFT_LOSS_COEF:-0.1}"
SFT_COEF_TAG="${TEACHER_PREFIX_SFT_LOSS_COEF/./p}"
export PREFIX_MODEL_OUTPUT_NAME_PREFIX="q4binst_q30binst2507_nothink_prefix128_sft${SFT_COEF_TAG}_opd_r4096_b96_n1_lr1e-5"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
# Same tokenizer/template/token IDs on both sides: use native Instruct chat
# templates and no cross-model EOS canonicalization bridge.
export REWARD_MODEL_INPUT_TOKENIZER="${REWARD_MODEL_INPUT_TOKENIZER:-}"
export STUDENT_CHAT_TEMPLATE_FILE="${STUDENT_CHAT_TEMPLATE_FILE:-}"
export TEACHER_PREFIX_MAX_LEN=128
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

export ENABLE_THINKING=False
export DATA_SHUFFLE=True
export DATA_SEED=42
export MAX_PROMPT_LENGTH=2048
export MAX_RESP_LENGTH=4096
export MAX_VAL_RESP_LENGTH=4096
export MINI_BATCH_SIZE=96
export N_RESPONSES=1
export LR="${LR:-1e-5}"
export TOTAL_EPOCHS=1
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
# Match the historical 4B-Instruct <- 30B-Instruct plain-OPD measurement.
export EVAL_MAX_TOKENS=9192
export EVAL_OUTPUT_DIR="${OPD_ROOT}/outputs/eval/q4binst_q30binst2507_nothink_prefix128"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_prefix128_sft${SFT_COEF_TAG}_opd_r4096_b96_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PREFIX_TRAIN_DATASET}" || {
    echo "Missing prefix dataset: ${PREFIX_TRAIN_DATASET}" >&2
    echo "Build it first: bash scripts/teacher_prefix/run_q4binst_q30binst2507_nothink_prefix128_4gpu.sh" >&2
    exit 1
}
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"

exec bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
