#!/usr/bin/env bash
set -euo pipefail

# Evaluate the pre-training student and teacher with the exact code benchmark
# configuration used for the plain-OPD checkpoint.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_IDS=${GPU_IDS:-0,1,2,3}
RUNNER="$ROOT/scripts/eval/run_eurus_code_benchmarks_4gpu.sh"

run_model() {
    local label=$1
    local model_dir=$2
    local run_name="eurus_code_${label}_n4_t1_p1"
    local log_file="$ROOT/logs/eval_${run_name}.log"

    [[ -f "$model_dir/config.json" ]] || { echo "Missing model config: $model_dir" >&2; exit 1; }
    echo "=== Evaluating $label: $model_dir ==="
    GPU_IDS="$GPU_IDS" \
        MODEL_DIR="$model_dir" \
        RUN_NAME="$run_name" \
        EVALPLUS_DATASETS="humaneval,mbpp" \
        RUN_LCB=1 \
        bash "$RUNNER" 2>&1 | tee "$log_file"
}

run_model student_base "$ROOT/../model/Qwen3-1.7B-Base"
run_model teacher_base "$ROOT/../model/Qwen3-4B-Base"
