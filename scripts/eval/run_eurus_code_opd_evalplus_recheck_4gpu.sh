#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate the plain-OPD checkpoint with the same batched EvalPlus path
# used for the student and teacher baselines. LCB is deliberately excluded.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_IDS=${GPU_IDS:-0,1,2,3}
MODEL_DIR=${MODEL_DIR:-"$ROOT/merged_models/q3b_q4b_thinking_eurus_code_plain_opd_r4096_b96_ep1_shuffle42_n1_lr1e-5_step261"}
RUN_NAME=${RUN_NAME:-q3b_q4b_thinking_eurus_code_plain_opd_r4096_b96_ep1_shuffle42_step261_evalplus_recheck_n4_t1_p1}

[[ -f "$MODEL_DIR/config.json" ]] || { echo "Missing model config: $MODEL_DIR" >&2; exit 1; }

GPU_IDS="$GPU_IDS" \
MODEL_DIR="$MODEL_DIR" \
RUN_NAME="$RUN_NAME" \
EVALPLUS_DATASETS="humaneval,mbpp" \
RUN_LCB=0 \
bash "$ROOT/scripts/eval/run_eurus_code_benchmarks_4gpu.sh"
