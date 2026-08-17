#!/usr/bin/env bash
# Qwen3-4B-Base IF-RLVR GRPO teacher. Does not submit an rjob.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
export PYTHONPATH="$ROOT/verl:$ROOT/third_party/open-instruct-ifrlvr${PYTHONPATH:+:$PYTHONPATH}"
export NLTK_DATA="$ROOT/third_party/nltk_data${NLTK_DATA:+:$NLTK_DATA}"
export VLLM_USE_FLASHINFER_SAMPLER=0

ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-"$ROOT/../model/Qwen3-4B-Base"}
TRAIN_DATASET=${TRAIN_DATASET:-"$ROOT/datasets/ifrlvr/ifrlvr_train_25056_balanced_constraints_seed42_verl.parquet"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-q4b_ifrlvr_balanced25k_grpo_r4096_n8_b96_ep2_shuffle42_lr5e-6}
MODEL_OUTPUT_NAME_PREFIX=${MODEL_OUTPUT_NAME_PREFIX:-"${EXPERIMENT_NAME}_lr5e-6"}
CKPT_PATH=${CKPT_PATH:-"$ROOT/checkpoint/$EXPERIMENT_NAME"}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-4096}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-96}
N_RESPONSES=${N_RESPONSES:-8}
LR=${LR:-5e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-131} # 0.5 epoch: floor(25056 / 96 / 2) = 130.5
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.88}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
ENABLE_THINKING=${ENABLE_THINKING:-True}
DATA_SEED=${DATA_SEED:-42}
REWARD_FUNCTION=${REWARD_FUNCTION:-"$ROOT/scripts/reward/ifrlvr.py"}

test -f "$ACTOR_MODEL_PATH/config.json"
test -f "$TRAIN_DATASET"
test -f "$REWARD_FUNCTION"
test -d "$ROOT/third_party/open-instruct-ifrlvr/open_instruct/IFEvalG"

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH))
PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_MODEL_LEN > 32768 ? MAX_MODEL_LEN : 32768))
EXPECTED_STEPS=$($PYTHON_BIN -c "import pyarrow.parquet as pq; n=pq.ParquetFile('${TRAIN_DATASET}').metadata.num_rows; print(n // ${MINI_BATCH_SIZE} * ${TOTAL_EPOCHS})")

echo "========== Qwen3-4B IF-RLVR GRPO =========="
echo "actor=$ACTOR_MODEL_PATH"
echo "dataset=$TRAIN_DATASET"
echo "Qwen thinking enabled=$ENABLE_THINKING; prompt/response=$MAX_PROMPT_LENGTH/$MAX_RESP_LENGTH"
echo "batch=$MINI_BATCH_SIZE; n=$N_RESPONSES; epochs=$TOTAL_EPOCHS; expected steps=$EXPECTED_STEPS"
echo "reward function=$REWARD_FUNCTION"
echo "save_freq=$SAVE_FREQ (0.5 epoch); lr=$LR; checkpoint=$CKPT_PATH"

ray stop --force || true
ray start --head
sleep 5

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
    reward_model.reward_manager=naive \
    custom_reward_function.path="$REWARD_FUNCTION" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=IFRLVRGRPO \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir="validation_log/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=False
