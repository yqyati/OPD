#!/usr/bin/env bash
# Code full-response SFT (7168) -> plain OPD.
# Student: Qwen3-1.7B-Base. Teacher: 4B Base GRPO step260.
# Run inside an already allocated four-GPU rjob. This script does not submit jobs.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl

export YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
export OPD_ROOT="${YANGQINGYU_ROOT}/workspace/OPD"
export MODEL_ROOT="${YANGQINGYU_ROOT}/model"
export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_TIMEOUT=7200
cd "${OPD_ROOT}"

PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
TRAIN_DATASET="datasets/eurus-2-code-verl/data/train-00000.parquet"
STUDENT_MODEL="${MODEL_ROOT}/Qwen3-1.7B-Base"
TEACHER_MODEL="${OPD_ROOT}/merged_models/q4b_eurus_code_binary_grpo_r7168_n8_b96_ep2_shuffle42_lr5e-6_step260"

# Stage 1 stores exact sampled token IDs. Stage 2 consumes those token IDs
# directly, so SFT and teacher rollout use the same native thinking template.
FULL_RESPONSE_DATASET="datasets/sft_teacher_response/q3b_q4bgrpo_step260teacher_eurus_code_full_response_7168_think.parquet"
SFT_DATASET="datasets/sft/q3b_q4bgrpo_step260teacher_eurus_code_full7168_think_pure_sft.parquet"
SFT_NAME="q3b_q4bgrpo_step260teacher_eurus_code_full7168_think_pure_sft_b96_lr1e-5"
SFT_CKPT="checkpoint/${SFT_NAME}"

OPD_NAME="q3b_q4bgrpo_step260teacher_eurus_code_full7168think_sftinit_plain_opd_r4096_b96_n1_lr1e-5"
OPD_CKPT="checkpoint/${OPD_NAME}"
OPD_MERGED="merged_models/${OPD_NAME}_step261"
EVAL_RUN_NAME="${OPD_NAME}_step261_n4_t0p2_p1_think"

for required in \
    "${TRAIN_DATASET}" \
    "${STUDENT_MODEL}/config.json" \
    "${TEACHER_MODEL}/config.json" \
    "scripts/sft/run_sharded_teacher_response_generation.sh" \
    "scripts/sft/make_teacher_prefix_sft_data.py" \
    "scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh" \
    "scripts/eval/run_eurus_code_benchmarks_4gpu.sh"; do
    test -e "${required}" || { echo "Missing required path: ${required}" >&2; exit 1; }
done

echo "========== Code full7168 SFT -> plain OPD =========="
echo "student=${STUDENT_MODEL}"
echo "teacher=${TEACHER_MODEL}"
echo "template=Qwen3 native; enable_thinking=True"
echo "teacher/SFT cap=7168; OPD training cap=4096; SFT/OPD batch=96; lr=1e-5"

if [[ ! -f "${FULL_RESPONSE_DATASET}" ]]; then
    echo "[stage 1/4] Generate exact teacher completions (thinking, max_new_tokens=7168)"
    RESPONSE_INPUT="${TRAIN_DATASET}" \
    RESPONSE_OUTPUT="${FULL_RESPONSE_DATASET}" \
    RESPONSE_TEACHER_MODEL="${TEACHER_MODEL}" \
    RESPONSE_GPU_GROUPS="0;1;2;3" \
    RESPONSE_TP=1 \
    RESPONSE_MAX_TOKENS=7168 \
    RESPONSE_MAX_MODEL_LEN=9216 \
    RESPONSE_BATCH_SIZE=32 \
    RESPONSE_TEMPERATURE=0.7 \
    RESPONSE_TOP_P=0.95 \
    RESPONSE_ENABLE_THINKING=True \
        bash scripts/sft/run_sharded_teacher_response_generation.sh
else
    echo "[stage 1/4] Reuse teacher completions: ${FULL_RESPONSE_DATASET}"
fi
test -f "${FULL_RESPONSE_DATASET}"

# Do not count the 7168 target tokens alone: MAX_LENGTH is prompt plus target.
if ! find "merged_models" -maxdepth 2 -type f -path "*/${SFT_NAME}_step*/config.json" -print -quit | grep -q .; then
    echo "[stage 2/4] Build exact-token 7168 full-SFT data"
    "${PYTHON_BIN}" scripts/sft/make_teacher_prefix_sft_data.py \
        --input "${FULL_RESPONSE_DATASET}" \
        --output "${SFT_DATASET}" \
        --tokenizer "${STUDENT_MODEL}" \
        --response-column teacher_response_text \
        --generated-token-ids-column teacher_response_token_ids \
        --finish-reason-column teacher_response_finish_reason \
        --max-length 9216 \
        --enable-thinking \
        --use-generated-token-ids

    SFT_ROWS=$("${PYTHON_BIN}" -c "import pyarrow.parquet as pq; print(pq.ParquetFile('${SFT_DATASET}').metadata.num_rows)")
    SFT_EXPECTED_STEPS=$((SFT_ROWS / 96))
    test "${SFT_EXPECTED_STEPS}" -ge 1 || { echo "SFT dataset is too small: ${SFT_ROWS} rows" >&2; exit 1; }
    echo "[stage 2/4] Full SFT rows=${SFT_ROWS}; expected steps=${SFT_EXPECTED_STEPS}"

    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="${SFT_DATASET}" \
        data.val_files="${SFT_DATASET}" \
        data.train_max_samples=-1 \
        data.val_max_samples=256 \
        data.multiturn.enable=False \
        data.custom_cls.path=scripts/sft/precomputed_token_sft_dataset.py \
        data.custom_cls.name=PrecomputedTokenSFTDataset \
        data.max_length=9216 \
        data.truncation=error \
        data.train_batch_size=96 \
        data.micro_batch_size_per_gpu=1 \
        +data.pad_mode=right \
        model.partial_pretrain="${STUDENT_MODEL}" \
        model.trust_remote_code=True \
        model.fsdp_config.model_dtype=bfloat16 \
        model.fsdp_config.offload_params=False \
        model.enable_gradient_checkpointing=True \
        use_remove_padding=True \
        optim.lr=1e-5 \
        trainer.default_local_dir="${SFT_CKPT}" \
        trainer.project_name=OnPolicyDistillation \
        trainer.experiment_name="${SFT_NAME}" \
        trainer.total_epochs=1 \
        trainer.seed=42 \
        trainer.save_freq=100 \
        trainer.test_freq=-1 \
        trainer.logger='["console","tensorboard"]' \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.max_ckpt_to_keep=1 \
        trainer.resume_mode=disable \
        trainer.checkpoint.save_contents='["hf_model"]' \
        trainer.checkpoint.load_contents='["hf_model"]'

    SFT_STEP=$(find "${SFT_CKPT}" -maxdepth 1 -type d -name 'global_step_*' \
        | sed -E 's#.*/global_step_([0-9]+)$#\1#' | sort -n | tail -1)
    test -n "${SFT_STEP}" || { echo "No SFT checkpoint found under ${SFT_CKPT}" >&2; exit 1; }
    test "${SFT_STEP}" -ge "${SFT_EXPECTED_STEPS}" || {
        echo "SFT stopped at step ${SFT_STEP}; expected ${SFT_EXPECTED_STEPS}." >&2
        exit 1
    }
    SFT_HF_DIR="${SFT_CKPT}/global_step_${SFT_STEP}/huggingface"
    SFT_MODEL_DIR="merged_models/${SFT_NAME}_step${SFT_STEP}"
    test -f "${SFT_HF_DIR}/config.json" || { echo "Missing SFT HF checkpoint: ${SFT_HF_DIR}" >&2; exit 1; }
    if [[ ! -f "${SFT_MODEL_DIR}/config.json" ]]; then
        cp -a "${SFT_HF_DIR}" "${SFT_MODEL_DIR}"
    fi
else
    echo "[stage 2/4] Reuse merged full-SFT model"
fi

SFT_MODEL_DIR=$(find "merged_models" -maxdepth 2 -type f -path "*/${SFT_NAME}_step*/config.json" -printf '%h\n' \
    | sort -V | tail -1)
test -n "${SFT_MODEL_DIR}" || { echo "Missing merged full-SFT model for ${SFT_NAME}" >&2; exit 1; }
test -f "${SFT_MODEL_DIR}/config.json"
echo "SFT warm-start model: ${SFT_MODEL_DIR}"

ray stop --force >/dev/null 2>&1 || true
if [[ ! -f "${OPD_MERGED}/config.json" ]]; then
    echo "[stage 3/4] Plain OPD from full7168 SFT model"
    RUN_MODE=plain \
    PLAIN_TRAIN_DATASET="${TRAIN_DATASET}" \
    PLAIN_TRAIN_DATASET_NAME="Eurus-RL-Code-Q3B-Q4BGRPO-Step260-Full7168ThinkSFT-PlainOPD" \
    PLAIN_MODEL_OUTPUT_NAME_PREFIX="${OPD_NAME}" \
    ACTOR_MODEL_PATH="${OPD_ROOT}/${SFT_MODEL_DIR}" \
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
    TEACHER_PREFIX_SFT_LOSS_COEF=0.0 \
    TEACHER_PREFIX_SOFT_KL_LOSS_COEF=0.0 \
    DISABLE_CUSTOM_REWARD_FUNCTION=True \
    EXTRA_PPO_ARGS="reward_model.reward_manager=batch" \
    SKIP_FINAL_EVAL=True \
    TEST_FILE="[\"${TRAIN_DATASET}\"]" \
    EXPERIMENT_NAME="${OPD_NAME}" \
    CKPT_PATH="${OPD_CKPT}" \
        bash scripts/train/run_qwen3_nothink_teacher_plain_then_prefix_4gpu.sh
else
    echo "[stage 3/4] Reuse merged plain-OPD model: ${OPD_MERGED}"
fi
test -f "${OPD_MERGED}/config.json"

echo "[stage 4/4] Evaluate full-SFT -> plain OPD: EvalPlus + official LCB v6"
MODEL_DIR="${OPD_ROOT}/${OPD_MERGED}" \
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
    bash scripts/eval/run_eurus_code_benchmarks_4gpu.sh

echo "Full7168 SFT -> plain OPD pipeline completed."
echo "SFT model: ${OPD_ROOT}/${SFT_MODEL_DIR}"
echo "OPD model: ${OPD_ROOT}/${OPD_MERGED}"
echo "EvalPlus: ${OPD_ROOT}/outputs/eval/code_benchmarks/${EVAL_RUN_NAME}/evalplus_batched"
