#!/usr/bin/env bash
# Fixed prefix=768 SFT + suffix OPD for Qwen3-1.7B-Base <- Qwen3-4B-Base-GRPO.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PREFIX_LENGTH=768
exec bash scripts/train/run_q3b_q4bgrpo_nothink_prefix256_opd_4gpu.sh
