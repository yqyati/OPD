#!/usr/bin/env bash
# Completion-aware prefix64 OPD for Qwen3-1.7B-Base <- 4B Base-GRPO step260.
# Uses the existing merged full teacher trajectories; no teacher re-generation.
# Run inside an already allocated four-GPU rjob. This script never submits jobs.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl

export YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
export OPD_ROOT="${YANGQINGYU_ROOT}/workspace/OPD"
export MODEL_ROOT="${YANGQINGYU_ROOT}/model"
export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "${OPD_ROOT}"

PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
SOURCE_DATASET="datasets/eurus-2-code-verl/data/train-00000.parquet"
FULL_RESPONSE_DATASET="datasets/sft_teacher_response/q3b_q4bgrpo_step260teacher_eurus_code_full_response_7168_think.parquet"
PREFIX_DATASET="datasets/teacher_prefix/q3b_q4bgrpo_step260teacher_eurus_code_prefix64_think.parquet"
STUDENT_MODEL="${MODEL_ROOT}/Qwen3-1.7B-Base"
TEACHER_MODEL="${OPD_ROOT}/merged_models/q4b_eurus_code_binary_grpo_r7168_n8_b96_ep2_shuffle42_lr5e-6_step260"
PREFIX_BUILDER="scripts/teacher_prefix/build_prefix_dataset_from_teacher_responses.py"
TRAIN_SCRIPT="scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh"
EVAL_SCRIPT="scripts/eval/run_eurus_code_benchmarks_4gpu.sh"

PREFIX_LENGTH=64
SFT_COEF=0.2
EXPERIMENT_NAME="q3b_q4bgrpo_step260teacher_eurus_code_prefix64_sft0p2_opd_r4096_b96_n1_lr1e-5"
CKPT_PATH="checkpoint/${EXPERIMENT_NAME}"
MERGED_MODEL="merged_models/${EXPERIMENT_NAME}_step261"
EVAL_RUN_NAME="${EXPERIMENT_NAME}_step261_n4_t0p2_p1_think"

for required in \
    "${SOURCE_DATASET}" \
    "${FULL_RESPONSE_DATASET}" \
    "${STUDENT_MODEL}/config.json" \
    "${TEACHER_MODEL}/config.json" \
    "${PREFIX_BUILDER}" \
    "${TRAIN_SCRIPT}" \
    "${EVAL_SCRIPT}"; do
    test -e "${required}" || { echo "Missing required path: ${required}" >&2; exit 1; }
done

echo "========== Prefix64 OPD: 1.7B Base <- 4B Base-GRPO step260 =========="
echo "source=${SOURCE_DATASET}"
echo "full_teacher_trajectories=${FULL_RESPONSE_DATASET}"
echo "student=${STUDENT_MODEL}"
echo "teacher=${TEACHER_MODEL}"
echo "template=Qwen3 native; enable_thinking=True"
echo "prefix=${PREFIX_LENGTH}; prefix-SFT coef=${SFT_COEF}; training response cap=4096; batch=96; lr=1e-5"

if [[ ! -f "${PREFIX_DATASET}" ]]; then
    echo "[stage 1/3] Build completion-aware exact-token prefix64 dataset"
    "${PYTHON_BIN}" "${PREFIX_BUILDER}" \
        --input "${FULL_RESPONSE_DATASET}" \
        --source "${SOURCE_DATASET}" \
        --output "${PREFIX_DATASET}" \
        --prefix-length "${PREFIX_LENGTH}"
else
    echo "[stage 1/3] Reuse prefix dataset: ${PREFIX_DATASET}"
fi
test -f "${PREFIX_DATASET}"

ray stop --force >/dev/null 2>&1 || true
if [[ ! -f "${MERGED_MODEL}/config.json" ]]; then
    echo "[stage 2/3] Train prefix64 SFT + suffix plain OPD, then merge"
    RUN_MODE=prefix \
    PREFIX_TRAIN_DATASET="${PREFIX_DATASET}" \
    PREFIX_TRAIN_DATASET_NAME="Eurus-RL-Code-Q3B-Q4BGRPO-Step260-Prefix64-Think-CompletionAware" \
    PREFIX_MODEL_OUTPUT_NAME_PREFIX="${EXPERIMENT_NAME}" \
    ACTOR_MODEL_PATH="${STUDENT_MODEL}" \
    REWARD_MODEL_PATH="${TEACHER_MODEL}" \
    REWARD_MODEL_INPUT_TOKENIZER="" \
    STUDENT_CHAT_TEMPLATE_FILE="" \
    ENABLE_THINKING=True \
    DATA_SHUFFLE=True \
    DATA_SEED=42 \
    MAX_PROMPT_LENGTH=2048 \
    MAX_RESP_LENGTH=4096 \
    MAX_VAL_RESP_LENGTH=4096 \
    MINI_BATCH_SIZE=96 \
    N_RESPONSES=1 \
    LR=1e-5 \
    TOTAL_EPOCHS=1 \
    TOTAL_TRAINING_STEPS=261 \
    LOG_PROB_TOP_K=16 \
    TOP_K_STRATEGY=only_stu \
    REWARD_WEIGHT_MODE=student_p \
    TEACHER_PREFIX_MAX_LEN="${PREFIX_LENGTH}" \
    TEACHER_PREFIX_SFT_LOSS_COEF="${SFT_COEF}" \
    TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0 \
    DISABLE_CUSTOM_REWARD_FUNCTION=True \
    EXTRA_PPO_ARGS="reward_model.reward_manager=batch" \
    SKIP_FINAL_EVAL=True \
    TEST_FILE="[\"${PREFIX_DATASET}\"]" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    CKPT_PATH="${CKPT_PATH}" \
        bash "${TRAIN_SCRIPT}"
else
    echo "[stage 2/3] Reuse merged prefix64 model: ${MERGED_MODEL}"
fi
test -f "${MERGED_MODEL}/config.json"

echo "[stage 3/3] Evaluate prefix64 OPD: thinking EvalPlus + official LCB v6"
MODEL_DIR="${OPD_ROOT}/${MERGED_MODEL}" \
RUN_NAME="${EVAL_RUN_NAME}" \
EVALPLUS_DATASETS="humaneval,mbpp" \
EVALPLUS_TEMPERATURE=0.2 \
EVALPLUS_ENABLE_THINKING=true \
EVAL_MAX_TOKENS=7168 \
RUN_LCB=1 \
GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3}" \
LCB_MODEL_NAME="Qwen3-1.7B-Thinking" \
LCB_N=10 \
LCB_TEMPERATURE=0.2 \
LCB_TOP_P=0.95 \
LCB_MAX_TOKENS=7168 \
    bash "${EVAL_SCRIPT}"

echo "Prefix64 OPD pipeline completed."
echo "Prefix data: ${OPD_ROOT}/${PREFIX_DATASET}"
echo "Merged model: ${OPD_ROOT}/${MERGED_MODEL}"
echo "EvalPlus: ${OPD_ROOT}/outputs/eval/code_benchmarks/${EVAL_RUN_NAME}/evalplus_batched"
