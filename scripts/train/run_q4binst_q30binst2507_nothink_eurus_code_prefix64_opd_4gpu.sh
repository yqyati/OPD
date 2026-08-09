#!/usr/bin/env bash
# Completion-aware fixed-prefix-64 code OPD:
# Qwen3-4B-Instruct-2507 <- Qwen3-30B-A3B-Instruct-2507, native no-think.
# This script trains, merges, and runs EvalPlus + LiveCodeBench locally; it
# never submits an rjob and never performs teacher rollout.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export RUN_MODE=prefix
export PREFIX_TRAIN_DATASET="${PREFIX_TRAIN_DATASET:-datasets/teacher_prefix/q4binst_q30binst2507_nothink_eurus_code_prefix64.parquet}"
export PREFIX_TRAIN_DATASET_NAME="Eurus-RL-Code-4BInst-30BInst-NoThink-Prefix64-CompletionAware"
export TEACHER_PREFIX_SFT_LOSS_COEF="${TEACHER_PREFIX_SFT_LOSS_COEF:-0.1}"
SFT_COEF_TAG="${TEACHER_PREFIX_SFT_LOSS_COEF/./p}"
export PREFIX_MODEL_OUTPUT_NAME_PREFIX="q4binst_q30binst2507_nothink_eurus_code_prefix64_sft${SFT_COEF_TAG}_opd_r4096_b96_n1_lr1e-5"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507}"
# Both models use the native Qwen3-Instruct no-think tokenizer/template/EOS
# contract; no Base<->Instruct bridge is used.
export REWARD_MODEL_INPUT_TOKENIZER=""
export STUDENT_CHAT_TEMPLATE_FILE=""
export CANONICAL_EOS_TOKEN_ID=""
export TEACHER_SOURCE_EOS_TOKEN_ID=""
export TEACHER_PREFIX_MAX_LEN=64
export TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0

export ENABLE_THINKING=False
export DATA_SHUFFLE=True
export DATA_SEED=42
# Preserve the 2048-token native code prompt budget and explicitly reserve the
# fixed 64-token teacher prefix at the handoff boundary.
export MAX_PROMPT_LENGTH=2112
export MAX_RESP_LENGTH=4096
export MAX_VAL_RESP_LENGTH=4096
export MINI_BATCH_SIZE=96
export N_RESPONSES=1
export LR="${LR:-1e-5}"
export TOTAL_EPOCHS=1
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p

# Pure token-OPD: use only teacher-derived rm_scores already attached to each
# rollout. ``batch`` returns those rm_scores immediately without decoding or
# executing code, so no CPU code verifier is launched during training.
export DISABLE_CUSTOM_REWARD_FUNCTION=True
export EXTRA_PPO_ARGS="reward_model.reward_manager=batch"
export SKIP_FINAL_EVAL=True
export TEST_FILE="[\"${PREFIX_TRAIN_DATASET}\"]"

RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-q4binst_q30binst2507_nothink_eurus_code_prefix64_sft${SFT_COEF_TAG}_opd_r4096_b96_n1_lr1e-5_${RUN_TAG}}"
export CKPT_PATH="${CKPT_PATH:-checkpoint/${EXPERIMENT_NAME}}"

test -f "${PREFIX_TRAIN_DATASET}"
test -f "${ACTOR_MODEL_PATH}/config.json"
test -f "${REWARD_MODEL_PATH}/config.json"
echo "Code pure token-OPD: reward_manager=batch; no CPU code verification"

bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh

MODEL_DIR="${OPD_ROOT}/merged_models/${PREFIX_MODEL_OUTPUT_NAME_PREFIX}_step261"
test -f "${MODEL_DIR}/config.json" || {
    echo "Expected merged model is missing: ${MODEL_DIR}" >&2
    exit 1
}

export MODEL_DIR
export RUN_NAME="${PREFIX_MODEL_OUTPUT_NAME_PREFIX}_step261_n4_t1_p1"
export EVALPLUS_DATASETS="humaneval,mbpp"
export RUN_LCB=1
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
# Registered LCB prompt/extraction style; --local_model_path supplies this
# experiment's actual local checkpoint.
export LCB_MODEL_NAME="Qwen3-4B-NonThinking"
exec bash scripts/eval/run_eurus_code_benchmarks_4gpu.sh
