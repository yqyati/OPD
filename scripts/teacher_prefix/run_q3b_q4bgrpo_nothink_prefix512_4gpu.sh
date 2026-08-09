#!/usr/bin/env bash
# Reuse the saved 4096-token teacher trajectories and slice fixed prefix=512.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PREFIX_LENGTH=512
export OUTPUT_DATASET="${OUTPUT_DATASET:-datasets/teacher_prefix/q3b_q4bgrpo_nothink_dapo_math17k_prefix512.parquet}"
exec bash scripts/teacher_prefix/run_q3b_q4bgrpo_nothink_prefix256_4gpu.sh
