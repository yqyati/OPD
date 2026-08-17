#!/usr/bin/env bash
set -euo pipefail

# Paper-matched code evaluation for the Eurus plain-OPD checkpoint.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
GPU_IDS=${GPU_IDS:-0,1,2,3}
EXPECTED_DP_WORLD_SIZE=${EXPECTED_DP_WORLD_SIZE:-4}
MODEL_DIR=${MODEL_DIR:-"$ROOT/merged_models/q3b_q4b_thinking_eurus_code_plain_opd_r4096_b96_ep1_shuffle42_n1_lr1e-5_step261"}
RUN_NAME=${RUN_NAME:-q3b_q4b_thinking_eurus_code_plain_opd_r4096_b96_ep1_shuffle42_step261_n4_t0p2_p1}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT/outputs/eval/code_benchmarks/$RUN_NAME"}
[[ "$MODEL_DIR" = /* ]] || MODEL_DIR="$ROOT/$MODEL_DIR"
[[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"
# Set to an empty string to skip EvalPlus and run only LCB.
EVALPLUS_DATASETS=${EVALPLUS_DATASETS-humaneval,mbpp}
RUN_LCB=${RUN_LCB:-1}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-16384}
EVALPLUS_ENABLE_THINKING=${EVALPLUS_ENABLE_THINKING:-false}
EVALPLUS_TEMPERATURE=${EVALPLUS_TEMPERATURE:-0.2}
EVALPLUS_TOP_P=${EVALPLUS_TOP_P:-1.0}
EVALPLUS_REQUEST_CHUNK_SIZE=${EVALPLUS_REQUEST_CHUNK_SIZE:-128}
LCB_MODEL_NAME=${LCB_MODEL_NAME:-Qwen3-1.7B-Thinking}
# LiveCodeBench sampling protocol with the project's thinking-enabled Qwen3
# template. The 7168-token cap matches the Base RL training response budget;
# it intentionally differs from the runner README's non-thinking default.
LCB_N=${LCB_N:-10}
LCB_TEMPERATURE=${LCB_TEMPERATURE:-0.2}
LCB_TOP_P=${LCB_TOP_P:-0.95}
LCB_MAX_TOKENS=${LCB_MAX_TOKENS:-7168}
# Set to 0 to rescore all existing LiveCodeBench generations without
# regenerating them. This is useful after a local evaluator-only change.
LCB_RESUME_EVALUATION=${LCB_RESUME_EVALUATION:-1}
EVALPLUS_DIR="$ROOT/paper/G-OPD/code_eval/coding/evalplus"
LCB_DIR="$ROOT/paper/G-OPD/code_eval/coding/LiveCodeBench"

# Ensure workers import the repository's patched EvalPlus provider (including
# the explicit Qwen3 enable_thinking argument), rather than an older package
# copy installed in the environment.
export PYTHONPATH="$EVALPLUS_DIR:$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
DP_WORLD_SIZE=${#GPU_LIST[@]}
[[ "$EXPECTED_DP_WORLD_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "EXPECTED_DP_WORLD_SIZE must be a positive integer; got $EXPECTED_DP_WORLD_SIZE" >&2
    exit 1
}
[[ "$DP_WORLD_SIZE" -eq "$EXPECTED_DP_WORLD_SIZE" ]] || {
    echo "This evaluation launcher requires exactly $EXPECTED_DP_WORLD_SIZE single-GPU DP workers; got GPU_IDS=$GPU_IDS" >&2
    exit 1
}
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
    local thinking_args=()
    if [[ "${EVALPLUS_ENABLE_THINKING,,}" == "true" || "${EVALPLUS_ENABLE_THINKING}" == "1" ]]; then
        thinking_args+=(--enable-thinking)
    fi

    mkdir -p "$result_dir"
    for rank in "${!GPU_LIST[@]}"; do
        gpu=${GPU_LIST[$rank]}
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/scripts/eval/batched_evalplus_codegen.py" \
            --dataset "$dataset" \
            --model "$MODEL_DIR" \
            --output "$result_dir/${dataset}_rank${rank}.jsonl" \
            --rank "$rank" \
            --world-size "$DP_WORLD_SIZE" \
            --n-samples 4 \
            --temperature "$EVALPLUS_TEMPERATURE" \
            --top-p "$EVALPLUS_TOP_P" \
            --max-tokens "$EVAL_MAX_TOKENS" \
            --request-chunk-size "$EVALPLUS_REQUEST_CHUNK_SIZE" \
            "${thinking_args[@]}" &
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
        --world-size "$DP_WORLD_SIZE" \
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

    # EvalPlus can leave a local Ray head behind. Clear it before launching
    # Independent single-GPU LiveCodeBench workers.
    ray stop --force >/dev/null 2>&1 || true
    unset RAY_ADDRESS

    local lcb_output_root="$LCB_DIR/lcb_outputs/$(basename "$MODEL_DIR")"
    local shard_root="$lcb_output_root/dp${DP_WORLD_SIZE}_shards"
    local merged_output="$lcb_output_root/Scenario.codegeneration_${LCB_N}_${LCB_TEMPERATURE}_dp${DP_WORLD_SIZE}_merged.json"
    local model_name
    model_name=$(basename "$MODEL_DIR")
    local rank gpu shard_model_dir shard_output
    local pids=()
    local worker_failed=0

    mkdir -p "$shard_root"
    for rank in "${!GPU_LIST[@]}"; do
        gpu=${GPU_LIST[$rank]}
        # The LCB runner keys resumable output by basename(local_model_path).
        # Include the checkpoint name so separate checkpoints cannot reuse a
        # previous checkpoint's DP shard generations.
        shard_model_dir="$shard_root/${model_name}_rank${rank}"
        shard_output="$LCB_DIR/lcb_outputs/$(basename "$shard_model_dir")/Scenario.codegeneration_${LCB_N}_${LCB_TEMPERATURE}.json"
        ln -sfn "$MODEL_DIR" "$shard_model_dir"
        (
            cd "$LCB_DIR"
            CUDA_VISIBLE_DEVICES="$gpu" LCB_TOKENIZER_PATH="$MODEL_DIR" PYTHONPATH="$LCB_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m lcb_runner.runner.main \
                --model "$LCB_MODEL_NAME" \
                --local_model_path "$shard_model_dir" \
                --trust_remote_code \
                --scenario codegeneration \
                --release_version v6 \
                --start_date 2025-02-01 \
                --end_date 2025-05-31 \
                --n "$LCB_N" \
                --temperature "$LCB_TEMPERATURE" \
                --top_p "$LCB_TOP_P" \
                --max_tokens "$LCB_MAX_TOKENS" \
                --tensor_parallel_size 1 \
                --multiprocess 1 \
                --num_process_evaluate 64 \
                --timeout 60 \
                --continue_existing \
                --shard-rank "$rank" \
                --shard-world-size "$DP_WORLD_SIZE"
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            worker_failed=1
        fi
    done
    [[ "$worker_failed" -eq 0 ]] || return 1

    local shard_outputs=()
    for rank in "${!GPU_LIST[@]}"; do
        shard_outputs+=("$LCB_DIR/lcb_outputs/${model_name}_rank${rank}/Scenario.codegeneration_${LCB_N}_${LCB_TEMPERATURE}.json")
    done
    "$PYTHON_BIN" "$ROOT/scripts/eval/merge_lcb_codegeneration_shards.py" \
        --output "$merged_output" "${shard_outputs[@]}"
    (
        cd "$LCB_DIR"
        PYTHONPATH="$LCB_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m lcb_runner.runner.custom_evaluator \
            --scenario codegeneration \
            --release_version v6 \
            --start_date 2025-02-01 \
            --end_date 2025-05-31 \
            --n "$LCB_N" \
            --temperature "$LCB_TEMPERATURE" \
            --num_process_evaluate 64 \
            --timeout 60 \
            --custom_output_file "$merged_output"
    )
}

echo "Model: $MODEL_DIR"
echo "GPUs: $GPU_IDS (DP=$DP_WORLD_SIZE, $DP_WORLD_SIZE independent single-GPU workers; tensor parallel is forbidden)"
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
