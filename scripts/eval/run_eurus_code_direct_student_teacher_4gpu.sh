#!/usr/bin/env bash
# Direct code evaluation only, no training:
#   student: Qwen3-4B-Instruct-2507
#   teacher: Qwen3-30B-A3B-Instruct-2507
# Each model is evaluated sequentially with the same four-GPU EvalPlus and
# LiveCodeBench v6 protocol used for the plain/prefix OPD code runs.
set -euo pipefail

source .env
cd "${OPD_ROOT}"

export EVALPLUS_DATASETS="humaneval,mbpp"
export RUN_LCB=1
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
# This is the registered LCB prompt/extraction style. --local_model_path in
# the benchmark launcher supplies the actual 4B or 30B local checkpoint.
export LCB_MODEL_NAME="Qwen3-4B-NonThinking"

run_direct_eval() {
    local label=$1
    local model_dir=$2

    test -f "${model_dir}/config.json" || {
        echo "Missing direct-evaluation model: ${model_dir}" >&2
        exit 1
    }

    export MODEL_DIR="${model_dir}"
    export RUN_NAME="q4binst_q30binst2507_nothink_eurus_code_direct_${label}_n4_t1_p1"
    # Ensure the benchmark launcher derives a model-specific fresh output path.
    unset OUTPUT_ROOT

    echo "================================================================"
    echo "Direct code evaluation: ${label}"
    echo "Model: ${MODEL_DIR}"
    echo "Protocol: EvalPlus HumanEval/MBPP + LiveCodeBench v6; n=4, T=1.0, top-p=1.0"
    echo "================================================================"
    bash scripts/eval/run_eurus_code_benchmarks_4gpu.sh
}

run_direct_eval "student_qwen3_4b_instruct2507" "${MODEL_ROOT}/Qwen3-4B-Instruct-2507"
run_direct_eval "teacher_qwen3_30b_a3b_instruct2507" "${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507"

echo "Completed direct code evaluation for both student and teacher."
