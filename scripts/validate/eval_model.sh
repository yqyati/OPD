#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL_DIR=""
DATA_DIR="${PROJECT_ROOT}/scripts/val/data"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/eval/justrl_eval_outputs_31744"
TASKS="AIME24,AIME25,AMC23"
N=16
MAX_TOKENS=31744
TEMPERATURE=0.7
TOP_P=0.95
GPUS="0,1,2,3"
DISABLE_THINKING=1

usage() {
  cat <<'EOF'
Usage: scripts/validate/eval_model.sh --model-dir PATH [options]

Required:
  --model-dir PATH       HuggingFace-format model directory to evaluate.

Options:
  --data-dir PATH        Evaluation data directory.
  --output-dir PATH      Directory where generations and grading are written.
  --tasks LIST           Comma-separated task list. Default: AIME24,AIME25,AMC23.
  --n N                  Number of samples per prompt. Default: 16.
  --max-tokens N         Max generation tokens. Default: 31744.
  --temperature FLOAT    Sampling temperature. Default: 0.7.
  --top-p FLOAT          Top-p sampling value. Default: 0.95.
  --gpus LIST            Comma-separated GPU ids. Default: 0,1,2,3.
  --enable-thinking      Do not pass --disable-thinking to gen_vllm.py.
  -h, --help             Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --enable-thinking)
      DISABLE_THINKING=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${TRACE:-1}" -eq 1 ]]; then
  set -x
fi

if [[ -z "${MODEL_DIR}" ]]; then
  echo "Missing required argument: --model-dir" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Missing model config: ${MODEL_DIR}/config.json" >&2
  exit 1
fi

EVAL_DIR="${OUTPUT_DIR}/$(basename "${MODEL_DIR}")"
THINKING_ARGS=()
if [[ "${DISABLE_THINKING}" -eq 1 ]]; then
  THINKING_ARGS+=(--disable-thinking)
fi

python scripts/val/eval/gen_vllm.py \
  --model "${MODEL_DIR}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --tasks "${TASKS}" \
  --n "${N}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --gpus "${GPUS}" \
  "${THINKING_ARGS[@]}"

python scripts/val/eval/grade.py \
  --eval-dir "${EVAL_DIR}"
