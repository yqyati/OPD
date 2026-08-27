#!/usr/bin/env bash
# Held-out google/IFEval evaluation of the merged 8B IF-RLVR model.
# Run inside a two-GPU rjob; this script does not submit an rjob itself.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl

export YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
export OPD_ROOT="${YANGQINGYU_ROOT}/workspace/OPD"
export PYTHONPATH="${OPD_ROOT}/verl:${OPD_ROOT}/third_party/open-instruct-ifrlvr:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
cd "${OPD_ROOT}"

PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
GPU_IDS=0,1
MODEL="merged_models/q8b_q30ba3b_ifrlvr_think_correct5917_sftinit_ifrlvr_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step516"
INPUT="datasets/eval/IFEval/ifeval_input_data.jsonl"
OUTPUT="outputs/eval/ifeval/q8b_q30ba3b_ifrlvr_think_sftinit_grpo_step516_think_t0_n1_m7168_dp2"
EVAL_SCRIPT="scripts/eval/eval_ifeval_g.py"

for required in "${MODEL}/config.json" "${INPUT}" "${EVAL_SCRIPT}" third_party/open-instruct-ifrlvr/open_instruct/IFEvalG; do
    test -e "${required}" || { echo "Missing required path: ${required}" >&2; exit 1; }
done

echo "================================================================"
echo "Held-out google/IFEval: merged 8B IF-RLVR step516"
echo "model=${MODEL}"
echo "data=${INPUT}; 541 official held-out prompts"
echo "generation: thinking=True; n=1; temperature=0.0; top-p=1.0; max_tokens=7168"
echo "topology: two independent single-GPU DP workers (${GPU_IDS}); TP=1"
echo "verifier: project open_instruct.IFEvalG registry; reports constraint-level and all-constraints prompt-level accuracy"
echo "output=${OUTPUT}"
echo "================================================================"

pids=()
rank=0
for gpu in ${GPU_IDS//,/ }; do
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --mode generate --model "${MODEL}" --input "${INPUT}" --output-dir "${OUTPUT}" \
        --rank "${rank}" --world-size 2 --temperature 0.0 --top-p 1.0 --max-tokens 7168 --max-model-len 9216 &
    pids+=("$!")
    rank=$((rank + 1))
done
failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done
[[ "${failed}" -eq 0 ]] || { echo "IFEval generation worker failed; rank JSONL files are resumable." >&2; exit 1; }

"${PYTHON_BIN}" "${EVAL_SCRIPT}" --mode score --input "${INPUT}" --output-dir "${OUTPUT}" --world-size 2
echo "IFEval completed: ${OUTPUT}/summary.json"
