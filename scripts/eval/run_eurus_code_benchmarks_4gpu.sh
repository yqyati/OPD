#!/usr/bin/env bash
set -euo pipefail

# Paper-matched code evaluation for the Eurus plain-OPD checkpoint.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MODEL_DIR=${MODEL_DIR:-"$ROOT/merged_models/q3b_q4b_thinking_eurus_code_plain_opd_r4096_b96_ep1_shuffle42_n1_lr1e-5_step261"}
RUN_NAME=${RUN_NAME:-q3b_q4b_thinking_eurus_code_plain_opd_r4096_b96_ep1_shuffle42_step261_n4_t1_p1}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT/outputs/eval/code_benchmarks/$RUN_NAME"}
# Set to an empty string to skip EvalPlus and run only LCB.
EVALPLUS_DATASETS=${EVALPLUS_DATASETS-humaneval,mbpp}
RUN_LCB=${RUN_LCB:-1}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-16384}
EVALPLUS_REQUEST_CHUNK_SIZE=${EVALPLUS_REQUEST_CHUNK_SIZE:-64}
LCB_MODEL_NAME=${LCB_MODEL_NAME:-Qwen3-1.7B-NonThinking}
# Set to 0 to rescore all existing LiveCodeBench generations without
# regenerating them. This is useful after a local evaluator-only change.
LCB_RESUME_EVALUATION=${LCB_RESUME_EVALUATION:-1}
EVALPLUS_DIR="$ROOT/paper/G-OPD/code_eval/coding/evalplus"
LCB_DIR="$ROOT/paper/G-OPD/code_eval/coding/LiveCodeBench"

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
TP=${#GPU_LIST[@]}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# This cluster image ships FlashInfer but not nvcc. Prevent vLLM from trying
# to JIT-compile its sampler; the training run used the PyTorch sampler too.
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export HUMANEVAL_OVERRIDE_PATH="$ROOT/datasets/evalplus/HumanEvalPlus-v0.1.10.jsonl.gz"
export MBPP_OVERRIDE_PATH="$ROOT/datasets/evalplus/MbppPlus-v0.2.0.jsonl.gz"

[[ -d "$MODEL_DIR" ]] || { echo "Missing merged model: $MODEL_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

run_evalplus() {
    local dataset=$1
    local result_dir="$OUTPUT_ROOT/evalplus_batched/$dataset"
    local merged_samples="$result_dir/${dataset}_merged.jsonl"
    local rank
    local gpu
    local pids=()
    local worker_failed=0

    mkdir -p "$result_dir"
    for rank in "${!GPU_LIST[@]}"; do
        gpu=${GPU_LIST[$rank]}
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/scripts/eval/batched_evalplus_codegen.py" \
            --dataset "$dataset" \
            --model "$MODEL_DIR" \
            --output "$result_dir/${dataset}_rank${rank}.jsonl" \
            --rank "$rank" \
            --world-size "$TP" \
            --n-samples 4 \
            --temperature 1.0 \
            --top-p 1.0 \
            --max-tokens "$EVAL_MAX_TOKENS" \
            --request-chunk-size "$EVALPLUS_REQUEST_CHUNK_SIZE" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            worker_failed=1
        fi
    done
    if [[ "$worker_failed" -ne 0 ]]; then
        echo "One or more $dataset generation workers failed. Existing shards are resumable; inspect their logs before retrying." >&2
        return 1
    fi

    "$PYTHON_BIN" "$ROOT/scripts/eval/batched_evalplus_codegen.py" \
        --merge \
        --dataset "$dataset" \
        --output "$merged_samples" \
        --world-size "$TP" \
        --n-samples 4

    (
        cd "$EVALPLUS_DIR"
        "$PYTHON_BIN" -m evalplus.evaluate \
            --dataset "$dataset" \
            --samples "$merged_samples" \
            --output_file "$result_dir/${dataset}_results.json" \
            --parallel 64 \
            --min-time-limit 10.0 \
            --gt-time-limit-factor 8.0
    )
}

run_lcb() {
    local data_dir="$LCB_DIR/code_generation_lite"
    local target="$data_dir/test6.jsonl"
    local source="$ROOT/datasets/livecodebench/v6/test6.jsonl"
    local continue_args=(--continue_existing)

    mkdir -p "$data_dir"
    if [[ -e "$target" && ! -L "$target" ]]; then
        echo "Refusing to replace existing LCB data file: $target" >&2
        exit 1
    fi
    ln -sfn "$source" "$target"
    if [[ "$LCB_RESUME_EVALUATION" == "1" ]]; then
        continue_args+=(--continue_existing_with_eval)
    fi

    if ! "$PYTHON_BIN" -c 'import pebble' >/dev/null 2>&1; then
        echo "LiveCodeBench requires the Python package 'pebble'. Install it in the active verl environment before launching evaluation." >&2
        return 1
    fi

    (
        cd "$LCB_DIR"
        LCB_TOKENIZER_PATH="$MODEL_DIR" PYTHONPATH="$LCB_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m lcb_runner.runner.main \
            --model "$LCB_MODEL_NAME" \
            --local_model_path "$MODEL_DIR" \
            --trust_remote_code \
            --scenario codegeneration \
            --release_version v6 \
            --start_date 2025-02-01 \
            --end_date 2025-05-31 \
            --n 4 \
            --temperature 1.0 \
            --top_p 1.0 \
            --max_tokens 16384 \
            --tensor_parallel_size "$TP" \
            --num_process_evaluate 64 \
            --timeout 60 \
            --evaluate \
            "${continue_args[@]}"
    )
}

echo "Model: $MODEL_DIR"
echo "GPUs: $GPU_IDS (tensor parallel: $TP)"
echo "Output: $OUTPUT_ROOT"
if [[ -n "$EVALPLUS_DATASETS" ]]; then
    IFS=',' read -r -a EVALPLUS_DATASET_LIST <<< "$EVALPLUS_DATASETS"
    for dataset in "${EVALPLUS_DATASET_LIST[@]}"; do
        run_evalplus "$dataset"
    done
fi
if [[ "$RUN_LCB" == "1" ]]; then
    run_lcb
fi
echo "Completed code evaluation. EvalPlus JSON: $OUTPUT_ROOT/evalplus_batched"
