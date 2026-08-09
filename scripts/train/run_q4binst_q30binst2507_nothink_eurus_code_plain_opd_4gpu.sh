#!/usr/bin/env bash
# Plain code OPD: Qwen3-4B-Instruct-2507 <- Qwen3-30B-A3B-Instruct-2507.
# Uses Eurus-RL-Code, native no-think Instruct templates, then runs the
# repository's four-way EvalPlus + LiveCodeBench code evaluation chain.
# This script never submits an rjob.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export RUN_MODE=plain
export PLAIN_TRAIN_DATASET="${PLAIN_TRAIN_DATASET:-datasets/eurus-2-code-verl/data/train-00000.parquet}"
export PLAIN_TRAIN_DATASET_NAME="Eurus-RL-Code-4BInst-30BInst-NoThink-PlainOPD"
export PLAIN_MODEL_OUTPUT_NAME_PREFIX="${PLAIN_MODEL_OUTPUT_NAME_PREFIX:-q4binst_q30binst2507_nothink_eurus_code_plain_opd_r4096_b96_n1_lr1e-5}"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
# Both sides are native Qwen3-Instruct.  No Base/Instruct template or EOS
# bridge is used, and the reward model consumes its native tokenizer IDs.
export REWARD_MODEL_INPUT_TOKENIZER="${REWARD_MODEL_INPUT_TOKENIZER:-}"
export STUDENT_CHAT_TEMPLATE_FILE="${STUDENT_CHAT_TEMPLATE_FILE:-}"
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""

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
export TEACHER_PREFIX_SFT_LOSS_COEF=0.0
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

# Pure token-OPD: retain the teacher-derived rm_scores already attached to
# each rollout and do not execute programs in a CPU verifier. The generic
# runner's final math evaluator is skipped; code evaluation follows below.
export DISABLE_CUSTOM_REWARD_FUNCTION=True
export EXTRA_PPO_ARGS="${EXTRA_PPO_ARGS:-reward_model.reward_manager=batch}"
export SKIP_FINAL_EVAL=True
export TEST_FILE="[\"${PLAIN_TRAIN_DATASET}\"]"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_eurus_code_plain_opd_r4096_b96_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PLAIN_TRAIN_DATASET}"
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"
echo "Code pure token-OPD: reward_manager=batch; no CPU code verification"

# This makes the model merge after the successful 261-step training run.
bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh

MODEL_DIR="${OPD_ROOT}/merged_models/${PLAIN_MODEL_OUTPUT_NAME_PREFIX}_step261"
test -f "${MODEL_DIR}/config.json" || {
    echo "Expected merged model is missing: ${MODEL_DIR}" >&2
    exit 1
}

# Matched unified code evaluation: n=4, temperature=1.0, top-p=1.0,
# max_tokens=16384, four independent EvalPlus generation shards, plus LCB v6.
export MODEL_DIR
export RUN_NAME="${PLAIN_MODEL_OUTPUT_NAME_PREFIX}_step261_n4_t1_p1"
export EVALPLUS_DATASETS="humaneval,mbpp"
export RUN_LCB=1
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
# LiveCodeBench's --model is a registered prompt/extraction style, not an
# experiment label.  The actual checkpoint is supplied by --local_model_path.
# This local Qwen3 no-think checkpoint must therefore use the registered style.
export LCB_MODEL_NAME="Qwen3-4B-NonThinking"
exec bash scripts/eval/run_eurus_code_benchmarks_4gpu.sh
