#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/OPD"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

CKPT_DIR=""
MERGED_DIR=""
BACKEND="fsdp"
FORCE_MERGE=0
EVAL_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/validate/eval_checkpoint.sh --ckpt-dir PATH --merged-dir PATH [eval options]

Required:
  --ckpt-dir PATH        FSDP actor checkpoint directory.
  --merged-dir PATH      Target HuggingFace-format model directory.

Options:
  --backend NAME         verl.model_merger backend. Default: fsdp.
  --force-merge          Run model_merger even if merged-dir/config.json exists.

All other options are forwarded to scripts/validate/eval_model.sh, for example:
  --output-dir PATH --max-tokens 31744 --gpus 0,1,2,3 --tasks AIME24,AIME25,AMC23
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --merged-dir)
      MERGED_DIR="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --force-merge)
      FORCE_MERGE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EVAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${TRACE:-1}" -eq 1 ]]; then
  set -x
fi

if [[ -z "${CKPT_DIR}" ]]; then
  echo "Missing required argument: --ckpt-dir" >&2
  usage >&2
  exit 2
fi

if [[ -z "${MERGED_DIR}" ]]; then
  echo "Missing required argument: --merged-dir" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Missing checkpoint dir: ${CKPT_DIR}" >&2
  exit 1
fi

if [[ "${FORCE_MERGE}" -eq 1 || ! -f "${MERGED_DIR}/config.json" ]]; then
  python -m verl.model_merger merge \
    --backend "${BACKEND}" \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${MERGED_DIR}"
fi

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "Missing model config after merge: ${MERGED_DIR}/config.json" >&2
  exit 1
fi

"${PROJECT_ROOT}/scripts/validate/eval_model.sh" \
  --model-dir "${MERGED_DIR}" \
  "${EVAL_ARGS[@]}"
