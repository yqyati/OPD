#!/usr/bin/env bash
# TMLR reviewer pool-size experiments on one allocated 8-GPU node.
#
# Panel A (fixed rank): evaluate the completed N128/r16 checkpoint and reuse
# the completed N256/r16 and N512/r16 checkpoints/predictions.
# Panel B (matched LoRA-pool budget): train/evaluate N128/r32 and N512/r8,
# and reuse N256/r16 as the shared center point. All three have N_L * rank=4096.
set -euo pipefail

YANGQINGYU_ROOT="/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu"
REPO_ROOT="${YANGQINGYU_ROOT}/workspace/lora_moe"
MODEL_PATH="${YANGQINGYU_ROOT}/model/Deepseek_v2_lite"
GSM8K_DATA="data/eval/gsm8k/test.parquet"
MATH500_DATA="data/eval/math500/test.jsonl"

FIXED_RESULT_ROOT="eval_results/tmlr_global_pool_sweep"
FIXED_LOG_ROOT="tmlr/logs/global_pool_sweep_rjob"
MATCHED_RESULT_ROOT="eval_results/tmlr_global_pool_sweep_matched_budget"
MATCHED_LOG_ROOT="tmlr/logs/global_pool_matched_budget_rjob"
TRAIN_LOG_ROOT="tmlr/logs"

FIXED_N128_CONFIG="examples/train_moe_lora_deepseek/rcp_global_pool_sweep_N128_rebuttal.yaml"
FIXED_N256_CONFIG="examples/train_moe_lora_deepseek/rcp_global_pool_sweep_N256_rebuttal.yaml"
FIXED_N512_CONFIG="examples/train_moe_lora_deepseek/rcp_global_pool_sweep_N512_rebuttal.yaml"
MATCHED_N128_CONFIG="examples/train_moe_lora_deepseek/rcp_global_pool_budget_N128_r32_rebuttal.yaml"
MATCHED_N512_CONFIG="examples/train_moe_lora_deepseek/rcp_global_pool_budget_N512_r8_rebuttal.yaml"

FIXED_N128_CKPT="saves/deepseek_v2_lite/moe_lora/rcp_global_pool_sweep_N128_rebuttal_math"
FIXED_N256_CKPT="saves/deepseek_v2_lite/moe_lora/rcp_global_pool_sweep_N256_rebuttal_math"
FIXED_N512_CKPT="saves/deepseek_v2_lite/moe_lora/rcp_global_pool_sweep_N512_rebuttal_math"
MATCHED_N128_CKPT="saves/deepseek_v2_lite/moe_lora/rcp_global_pool_budget_N128_r32_rebuttal_math"
MATCHED_N512_CKPT="saves/deepseek_v2_lite/moe_lora/rcp_global_pool_budget_N512_r8_rebuttal_math"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd "${REPO_ROOT}"

command -v llamafactory-cli >/dev/null 2>&1 || {
  echo "llamafactory-cli is unavailable after activating the base Conda environment." >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29521}"
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-${YANGQINGYU_ROOT}/.cache/huggingface_tmlr}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export PYTHONPATH="${REPO_ROOT}/rebuttal:${REPO_ROOT}/src:${PYTHONPATH:-}"
export MOE2LORA_DISABLE_DEEPSPEED_IMPORT=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER:-root}-triton-cache}"

mkdir -p \
  "${TRITON_CACHE_DIR}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${FIXED_RESULT_ROOT}" \
  "${FIXED_LOG_ROOT}" \
  "${MATCHED_RESULT_ROOT}" \
  "${MATCHED_LOG_ROOT}" \
  "${TRAIN_LOG_ROOT}"

for required in \
    "${MODEL_PATH}/config.json" \
    "${GSM8K_DATA}" \
    "${MATH500_DATA}" \
    "eval_scripts/eval_gsm8k.py" \
    "eval_scripts/eval_math500.py" \
    "${FIXED_N128_CONFIG}" \
    "${FIXED_N256_CONFIG}" \
    "${FIXED_N512_CONFIG}" \
    "${MATCHED_N128_CONFIG}" \
    "${MATCHED_N512_CONFIG}"; do
  test -e "${required}" || { echo "Missing required path: ${required}" >&2; exit 1; }
done

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
test "${GPU_COUNT}" -eq "${NPROC_PER_NODE}" || {
  echo "Expected ${NPROC_PER_NODE} visible GPUs, but PyTorch sees ${GPU_COUNT}." >&2
  exit 1
}

python - "${MODEL_PATH}" \
  "${FIXED_N128_CONFIG}" "${FIXED_N256_CONFIG}" "${FIXED_N512_CONFIG}" \
  "${MATCHED_N128_CONFIG}" "${MATCHED_N512_CONFIG}" <<'PY'
import sys

import accelerate.utils.other as other
import torch
import transformers
import yaml

model_path = sys.argv[1]
fixed_paths = sys.argv[2:5]
matched_paths = sys.argv[5:7]


def load(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


configs = [load(path) for path in fixed_paths + matched_paths]
reference = configs[1]
common_keys = (
    "stage", "finetuning_type", "moe_lora_n_groups", "moe_lora_top_k",
    "moe_lora_pool_share", "moe_lora_w_share", "moe_lora_target_layers",
    "moe_lora_hidden_bottleneck_dim", "dataset", "template", "cutoff_len",
    "max_samples", "per_device_train_batch_size", "gradient_accumulation_steps",
    "learning_rate", "num_train_epochs", "lr_scheduler_type", "warmup_ratio",
    "bf16", "gradient_checkpointing", "ddp_broadcast_buffers", "seed",
)

for path, config in zip(fixed_paths + matched_paths, configs):
    if config.get("model_name_or_path") != model_path:
        raise SystemExit(f"{path}: unexpected model_name_or_path")
    mismatches = [key for key in common_keys if config.get(key) != reference.get(key)]
    if mismatches:
        raise SystemExit(f"{path}: controlled settings differ: {mismatches}")

for path, config, expected_size in zip(fixed_paths, configs[:3], (128, 256, 512)):
    expected = (expected_size, 16, 32)
    actual = (
        config.get("moe_lora_n_experts"),
        config.get("moe_lora_rank"),
        config.get("moe_lora_alpha"),
    )
    if actual != expected:
        raise SystemExit(f"{path}: expected fixed-rank tuple {expected}, got {actual}")

for path, config, expected in zip(
    matched_paths,
    configs[3:],
    ((128, 32, 64), (512, 8, 16)),
):
    actual = (
        config.get("moe_lora_n_experts"),
        config.get("moe_lora_rank"),
        config.get("moe_lora_alpha"),
    )
    if actual != expected:
        raise SystemExit(f"{path}: expected matched-budget tuple {expected}, got {actual}")
    if actual[0] * actual[1] != 4096 or actual[2] / actual[1] != 2:
        raise SystemExit(f"{path}: N_L*rank or alpha/rank is not controlled")
    if config.get("ddp_find_unused_parameters") is not True:
        raise SystemExit(f"{path}: sparse DDP requires ddp_find_unused_parameters=true")

print("========== TMLR global-pool experiments ==========")
print(f"torch={torch.__version__}; torch CUDA={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"visible GPUs={torch.cuda.device_count()}")
print(f"DeepSpeed visible to Accelerate={other.is_deepspeed_available()}")
if other.is_deepspeed_available():
    raise SystemExit("DeepSpeed must be disabled for this ordinary-DDP pipeline")
print("[preflight] Fixed-rank and matched-pool-budget controls verified.")
PY

checkpoint_complete() {
  local path="$1"
  [[ -f "${path}/moe_lora_config.json" ]] && \
    { [[ -f "${path}/moe_lora_state.safetensors" ]] || \
      [[ -f "${path}/moe_lora_state.safetensors.index.json" ]]; }
}

jsonl_complete() {
  local path="$1"
  local expected_rows="$2"
  [[ -f "${path}" ]] && [[ "$(wc -l < "${path}")" -eq "${expected_rows}" ]]
}

train_one() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local log_file="$4"

  if checkpoint_complete "${checkpoint}"; then
    echo "[train ${label}] Reuse completed checkpoint: ${checkpoint}"
    return
  fi

  echo "[train ${label}] config=${config}"
  echo "[train ${label}] output=${checkpoint}"
  llamafactory-cli train "${config}" 2>&1 | tee "${log_file}"
  checkpoint_complete "${checkpoint}" || {
    echo "Training returned without a complete checkpoint: ${checkpoint}" >&2
    exit 1
  }
}

evaluate_one() {
  local label="$1"
  local checkpoint="$2"
  local result_dir="$3"
  local log_root="$4"

  checkpoint_complete "${checkpoint}" || {
    echo "Missing completed checkpoint for ${label}: ${checkpoint}" >&2
    exit 1
  }
  mkdir -p "${result_dir}" "${log_root}"

  if jsonl_complete "${result_dir}/gsm8k.jsonl" 1319; then
    echo "[eval ${label}] Reuse complete GSM8K predictions"
  else
    echo "[eval ${label}] GSM8K"
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
      eval_scripts/eval_gsm8k.py \
      --base_model "${MODEL_PATH}" \
      --adapter_path "${checkpoint}" \
      --data_path "${GSM8K_DATA}" \
      --batch_size "${EVAL_BATCH_SIZE:-64}" \
      --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
      --save_path "${result_dir}/gsm8k.jsonl" \
      2>&1 | tee "${log_root}/${label}_gsm8k.log"
  fi

  if jsonl_complete "${result_dir}/math500.jsonl" 500; then
    echo "[eval ${label}] Reuse complete MATH-500 predictions"
  else
    echo "[eval ${label}] MATH-500"
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
      eval_scripts/eval_math500.py \
      --base_model "${MODEL_PATH}" \
      --adapter_path "${checkpoint}" \
      --data_path "${MATH500_DATA}" \
      --batch_size "${EVAL_BATCH_SIZE:-64}" \
      --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
      --save_path "${result_dir}/math500.jsonl" \
      2>&1 | tee "${log_root}/${label}_math500.log"
  fi
}

write_summary() {
  local output_root="$1"
  shift
  python - "${output_root}" "$@" <<'PY'
import json
import os
import sys

output_root, *specs = sys.argv[1:]
summary = {}
for spec in specs:
    label, result_dir = spec.split("=", 1)
    summary[label] = {}
    for benchmark in ("gsm8k", "math500"):
        path = os.path.join(result_dir, f"{benchmark}.jsonl")
        with open(path, encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        correct = sum(bool(row["correct"]) for row in rows)
        summary[label][benchmark] = {
            "accuracy": correct / len(rows),
            "correct": correct,
            "total": len(rows),
        }

os.makedirs(output_root, exist_ok=True)
output = os.path.join(output_root, "summary.json")
with open(output, "w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps(summary, indent=2))
print(f"Saved summary to {output}")
PY
}

echo "[stage 1/4] Complete fixed-rank evaluation (only N128/r16 is missing)"
evaluate_one "N128_r16" "${FIXED_N128_CKPT}" "${FIXED_RESULT_ROOT}/N128" "${FIXED_LOG_ROOT}"
evaluate_one "N256_r16" "${FIXED_N256_CKPT}" "${FIXED_RESULT_ROOT}/N256" "${FIXED_LOG_ROOT}"
evaluate_one "N512_r16" "${FIXED_N512_CKPT}" "${FIXED_RESULT_ROOT}/N512" "${FIXED_LOG_ROOT}"
write_summary "${FIXED_RESULT_ROOT}" \
  "128=${FIXED_RESULT_ROOT}/N128" \
  "256=${FIXED_RESULT_ROOT}/N256" \
  "512=${FIXED_RESULT_ROOT}/N512"

echo "[stage 2/4] Train matched-budget endpoints; reuse N256/r16"
checkpoint_complete "${FIXED_N256_CKPT}" || {
  echo "The shared N256/r16 center checkpoint is incomplete: ${FIXED_N256_CKPT}" >&2
  exit 1
}
echo "[train N256_r16] Reuse shared center checkpoint: ${FIXED_N256_CKPT}"
train_one "N128_r32" "${MATCHED_N128_CONFIG}" "${MATCHED_N128_CKPT}" \
  "${TRAIN_LOG_ROOT}/global_pool_budget_N128_r32_train.log"
train_one "N512_r8" "${MATCHED_N512_CONFIG}" "${MATCHED_N512_CKPT}" \
  "${TRAIN_LOG_ROOT}/global_pool_budget_N512_r8_train.log"

echo "[stage 3/4] Evaluate matched-budget endpoints; reuse N256/r16 predictions"
evaluate_one "N128_r32" "${MATCHED_N128_CKPT}" \
  "${MATCHED_RESULT_ROOT}/N128_r32" "${MATCHED_LOG_ROOT}"
evaluate_one "N512_r8" "${MATCHED_N512_CKPT}" \
  "${MATCHED_RESULT_ROOT}/N512_r8" "${MATCHED_LOG_ROOT}"
jsonl_complete "${FIXED_RESULT_ROOT}/N256/gsm8k.jsonl" 1319 || {
  echo "Shared N256/r16 GSM8K predictions are incomplete." >&2
  exit 1
}
jsonl_complete "${FIXED_RESULT_ROOT}/N256/math500.jsonl" 500 || {
  echo "Shared N256/r16 MATH-500 predictions are incomplete." >&2
  exit 1
}

echo "[stage 4/4] Write the matched-budget three-point summary"
write_summary "${MATCHED_RESULT_ROOT}" \
  "128=${MATCHED_RESULT_ROOT}/N128_r32" \
  "256=${FIXED_RESULT_ROOT}/N256" \
  "512=${MATCHED_RESULT_ROOT}/N512_r8"

echo "TMLR fixed-rank and matched-pool-budget pipelines completed."
