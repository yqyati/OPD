#!/usr/bin/env bash
# Finish the remaining 8B Code / IF-RLVR work without repeating completed training.
# 1) Merge the completed 8B IF-RLVR GRPO step516 FSDP checkpoint.
# 2) Run a clean two-DP-worker evaluation of the completed 8B Code GRPO step523.
# This launcher does not submit an rjob itself.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl

export YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
export OPD_ROOT="${YANGQINGYU_ROOT}/workspace/OPD"
export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_TIMEOUT=7200
cd "${OPD_ROOT}"

PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
GPU_IDS=0,1

CODE_MODEL="merged_models/q8b_q30ba3b_eurus_code_think_correct_max7168_sftinit_code_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step523"
CODE_EVAL_NAME="q8b_q30ba3b_eurus_code_think_correct_max7168_sftinit_code_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_code_think_dp2_evalplusn4_t0p2_lcbn10_t0p2"
CODE_EVAL_ROOT="outputs/eval/code_benchmarks/${CODE_EVAL_NAME}"

IFRLVR_NAME="q8b_q30ba3b_ifrlvr_think_correct5917_sftinit_ifrlvr_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6"
IFRLVR_ACTOR_CKPT="checkpoint/${IFRLVR_NAME}/global_step_516/actor"
IFRLVR_MERGED="merged_models/${IFRLVR_NAME}_step516"

for required in \
    "${CODE_MODEL}/config.json" \
    "${IFRLVR_ACTOR_CKPT}" \
    "scripts/eval/run_eurus_code_benchmarks_4gpu.sh" \
    "scripts/eval/batched_evalplus_codegen.py" \
    "scripts/eval/merge_lcb_codegeneration_shards.py"; do
    test -e "${required}" || { echo "Missing required path: ${required}" >&2; exit 1; }
done

echo "================================================================"
echo "Finish remaining 8B Code + IF-RLVR work"
echo "IF-RLVR: merge completed FSDP actor checkpoint step516 -> ${IFRLVR_MERGED}"
echo "Code: evaluate completed GRPO model step523 with two independent DP workers (${GPU_IDS}), TP=1"
echo "Code EvalPlus: thinking=True, n=4, T=0.2, top-p=1.0, max_tokens=16384"
echo "Code LiveCodeBench: n=10, T=0.2, top-p=0.95, max_tokens=7168"
echo "Code evaluation output (new DP2 namespace): ${CODE_EVAL_ROOT}"
echo "================================================================"

if [[ -f "${IFRLVR_MERGED}/config.json" ]]; then
    echo "[stage 1/2] Reuse merged 8B IF-RLVR model: ${IFRLVR_MERGED}"
else
    test ! -e "${IFRLVR_MERGED}" || {
        echo "Found incomplete IF-RLVR merge directory: ${IFRLVR_MERGED}; inspect it before retrying." >&2
        exit 1
    }
    echo "[stage 1/2] Merge 8B IF-RLVR GRPO checkpoint: ${IFRLVR_ACTOR_CKPT}"
    IFRLVR_STAGE="${IFRLVR_MERGED}.tmp.$$"
    "${PYTHON_BIN}" -m verl.model_merger merge --backend fsdp --local_dir "${IFRLVR_ACTOR_CKPT}" --target_dir "${IFRLVR_STAGE}"
    test -f "${IFRLVR_STAGE}/config.json" || { echo "Incomplete IF-RLVR merge: ${IFRLVR_STAGE}" >&2; exit 1; }
    mv "${IFRLVR_STAGE}" "${IFRLVR_MERGED}"
fi

echo "[stage 2/2] Clean 2-GPU Code evaluation"
GPU_IDS="${GPU_IDS}" \
EXPECTED_DP_WORLD_SIZE=2 \
MODEL_DIR="${CODE_MODEL}" \
RUN_NAME="${CODE_EVAL_NAME}" \
OUTPUT_ROOT="${CODE_EVAL_ROOT}" \
EVALPLUS_ENABLE_THINKING=true \
EVALPLUS_TEMPERATURE=0.2 \
EVALPLUS_TOP_P=1.0 \
EVAL_MAX_TOKENS=16384 \
LCB_N=10 \
LCB_TEMPERATURE=0.2 \
LCB_TOP_P=0.95 \
LCB_MAX_TOKENS=7168 \
bash scripts/eval/run_eurus_code_benchmarks_4gpu.sh

echo "Completed remaining work."
echo "Merged 8B IF-RLVR model: ${IFRLVR_MERGED}"
echo "8B Code DP2 evaluation: ${CODE_EVAL_ROOT}"
