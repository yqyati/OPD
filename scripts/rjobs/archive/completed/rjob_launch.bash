#!/usr/bin/env bash
# Eurus-Code binary GRPO initialized from the code-thinking SFT step-156 model.
# Run only inside an already allocated four-GPU rjob; this script never submits a job.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl

export YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
export OPD_ROOT="$YANGQINGYU_ROOT/workspace/OPD"
export PYTHONPATH="$OPD_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0

# Four independent single-GPU DP workers. Do not use tensor parallelism.
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
cd "$OPD_ROOT"

TRAIN_DATASET="datasets/eurus-2-code-verl/data/train-00000.parquet"
ACTOR_MODEL_PATH="checkpoint/q4b_q30ba3b_eurus_code_think_correct7529_sft_b96_lr1e-5_ep2/global_step_156/huggingface"
TRAIN_SCRIPT="scripts/train/run_qwen3_4b_eurus_code_grpo_4gpu.sh"

EXPERIMENT_NAME="q4b_q30ba3b_eurus_code_think_correct7529_sft_step156_binary_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6"
CKPT_PATH="checkpoint/$EXPERIMENT_NAME"
MODEL_OUTPUT_NAME_PREFIX="$EXPERIMENT_NAME"

for required in "$TRAIN_DATASET" "$ACTOR_MODEL_PATH/config.json" "$TRAIN_SCRIPT"; do
    test -f "$required" || { echo "Missing required file: $required" >&2; exit 2; }
done

echo "========== Qwen3-4B Code-Thinking SFT step-156 -> Eurus-Code GRPO =========="
echo "actor initialization: $ACTOR_MODEL_PATH"
echo "dataset: $TRAIN_DATASET"
echo "topology: 4 GPUs, 4 independent single-GPU DP workers, tensor parallel size 1"
echo "prompt/response: 2048/7168; group n: 8; batch: 48; epochs: 1; lr: 5e-6; seed: 42"
echo "checkpoint: $CKPT_PATH"

ACTOR_MODEL_PATH="$ACTOR_MODEL_PATH" \
TRAIN_DATASET="$TRAIN_DATASET" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_OUTPUT_NAME_PREFIX="$MODEL_OUTPUT_NAME_PREFIX" \
CKPT_PATH="$CKPT_PATH" \
N_GPUS_PER_NODE=4 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESP_LENGTH=7168 \
MINI_BATCH_SIZE=48 \
N_RESPONSES=8 \
LR=5e-6 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=261 \
ENABLE_THINKING=True \
DATA_SEED=42 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=65536 \
    bash "$TRAIN_SCRIPT"

echo "Eurus-Code GRPO completed. Checkpoints:"
find "$CKPT_PATH" -maxdepth 1 -type d -name 'global_step_*' | sort -V
