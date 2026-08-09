#!/usr/bin/env bash
# Reuse the saved 4096-token teacher trajectories and slice fixed prefix=1024.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PREFIX_LENGTH=1024
export OUTPUT_DATASET="${OUTPUT_DATASET:-datasets/teacher_prefix/q3b_q4bgrpo_nothink_dapo_math17k_prefix1024.parquet}"
exec bash scripts/teacher_prefix/run_q3b_q4bgrpo_nothink_prefix256_4gpu.sh
