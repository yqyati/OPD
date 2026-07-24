#!/usr/bin/env bash
set -euo pipefail

# Standalone code GRPO: no teacher model, token OPD, or teacher-prefix terms.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
# This repo keeps the importable ``verl`` package under OPD/verl rather than
# installing it into the environment. Make the standalone launcher explicit.
export PYTHONPATH="$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
# FlashInfer is installed on this image but nvcc is not. Its sampler JIT would
# fail during vLLM's profile pass, so use vLLM's PyTorch-native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-"$ROOT/../model/Qwen3-4B-Base"}
TRAIN_DATASET=${TRAIN_DATASET:-"$ROOT/datasets/eurus-2-code-verl/data/train-00000.parquet"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-q4b_eurus_code_binary_grpo_r4096_n8_b96_ep1_shuffle42}
MODEL_OUTPUT_NAME_PREFIX=${MODEL_OUTPUT_NAME_PREFIX:-"${EXPERIMENT_NAME}_lr1e-6"}
CKPT_PATH=${CKPT_PATH:-"$ROOT/checkpoint/${EXPERIMENT_NAME}_$(date +%Y-%m-%d_%H-%M-%S)"}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-4096}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-96}
N_RESPONSES=${N_RESPONSES:-8}
LR=${LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
CODE_REWARD_WORKERS=${CODE_REWARD_WORKERS:-64}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.9}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
ENABLE_THINKING=${ENABLE_THINKING:-True}
DATA_SEED=${DATA_SEED:-42}

test -f "$ACTOR_MODEL_PATH/config.json"
test -f "$TRAIN_DATASET"

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH))
PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_MODEL_LEN > 32768 ? MAX_MODEL_LEN : 32768))
EXPECTED_STEPS=$($PYTHON_BIN -c "import pyarrow.parquet as pq; n=pq.ParquetFile('${TRAIN_DATASET}').metadata.num_rows; print(max(1, n // ${MINI_BATCH_SIZE} * ${TOTAL_EPOCHS}))")
MIN_SUCCESS_STEP=$($PYTHON_BIN -c "import math; print(math.ceil(${EXPECTED_STEPS} * 0.9))")

echo "========== Qwen3-4B Eurus Code Binary GRPO =========="
echo "actor: $ACTOR_MODEL_PATH"
echo "dataset: $TRAIN_DATASET"
echo "GPUs: $N_GPUS_PER_NODE; prompt/response: $MAX_PROMPT_LENGTH/$MAX_RESP_LENGTH; group n: $N_RESPONSES"
echo "batch: $MINI_BATCH_SIZE; expected steps: $EXPECTED_STEPS; binary verifier workers: $CODE_REWARD_WORKERS"
echo "checkpoint: $CKPT_PATH"

ray stop --force || true
ray start --head
sleep 5

set +e
# Validation is disabled below, but main_ppo still constructs a validation
# dataset. Reuse Eurus Code instead of inheriting the math default.
$PYTHON_BIN -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_outcome_weight=1.0 \
    data.shuffle=True \
    data.seed=$DATA_SEED \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TRAIN_DATASET" \
    data.train_batch_size=$MINI_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.actor.checkpoint.save_contents=[model] \
    actor_rollout_ref.actor.checkpoint.load_contents=[model] \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward_model.enable=False \
    reward_model.reward_manager=parallel_grpo_code \
    +reward_model.reward_kwargs.num_processes=$CODE_REWARD_WORKERS \
    custom_reward_function=null \
    trainer.val_before_train=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=CodeGRPO \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir="validation_log/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=False
TRAIN_EXIT=$?
set -e

echo "Training exit code: $TRAIN_EXIT"
echo "Checkpoint path: $CKPT_PATH"
if [[ "$TRAIN_EXIT" -ne 0 ]]; then
    echo "Training failed before checkpoint creation; skip checkpoint discovery and merge." >&2
    exit "$TRAIN_EXIT"
fi

STEP=${STEP:-latest}
if [[ "$STEP" == latest ]]; then
    STEP=$(find "$CKPT_PATH" -maxdepth 1 -type d -name 'global_step_*' | sed -E 's#.*/global_step_([0-9]+)$#\1#' | sort -n | tail -1)
fi
if [[ -z "$STEP" || "$STEP" -lt "$MIN_SUCCESS_STEP" ]]; then
    echo "No sufficiently complete checkpoint found; skip merge." >&2
    exit "$TRAIN_EXIT"
fi

CKPT_DIR="$CKPT_PATH/global_step_${STEP}/actor"
MERGED_MODEL_DIR="$ROOT/merged_models/${MODEL_OUTPUT_NAME_PREFIX}_step${STEP}"
test -d "$CKPT_DIR"
$PYTHON_BIN -m verl.model_merger merge --backend fsdp --local_dir "$CKPT_DIR" --target_dir "$MERGED_MODEL_DIR"
echo "Merged model: $MERGED_MODEL_DIR"
exit "$TRAIN_EXIT"
