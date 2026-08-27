#!/usr/bin/env bash
# Same-origin Prefix-256 MOPD: General-SFT ckp1 student and three same-origin RL teachers.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl
source /mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/OPD/.env

export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_TIMEOUT=7200
cd "${OPD_ROOT}"

PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
RAY_BIN=/root/miniconda3/envs/verl/bin/ray
STUDENT_GPUS=5
EXPECTED_STEP=484
RUN_NAME=q8b_sameorigin_prefix256_mopd_r7168_n1_b140_mb40_ep1_lr1e-6
CHECKPOINT_ROOT="checkpoint/mopd/${RUN_NAME}"
ACTOR_CHECKPOINT="${CHECKPOINT_ROOT}/global_step_${EXPECTED_STEP}/actor"
MERGED_MODEL="merged_models/${RUN_NAME}_step${EXPECTED_STEP}"
PREFIX_DATASET=datasets/mopd/q8b_sameorigin_mopd_math_code_instruct_teacher_prefix256_think.parquet
STUDENT_MODEL="${OPD_ROOT}/checkpoint/q8b_general_sft_math2k_code2k_instruct2k_seed42_b96_lr1e-5_ep2/global_step_62/huggingface"
MATH_TEACHER="${OPD_ROOT}/merged_models/q8b_math_grpo_from_general_sft_ep1_r7168_n8_b48_ep1_shuffle42_lr5e-6_step373"
CODE_TEACHER="${OPD_ROOT}/merged_models/q8b_code_grpo_from_general_sft_ep1_r7168_n8_b48_ep1_shuffle42_lr5e-6_step523"
INSTRUCT_TEACHER="${OPD_ROOT}/merged_models/q8b_instruct_grpo_from_general_sft_ep1_r7168_n8_b48_ep1_shuffle42_lr5e-6_step516"
HYDRA_ARGS=()
[[ "${MOPD_CONFIG_ONLY:-0}" == "1" ]] && HYDRA_ARGS=(--cfg job)

cleanup() { "${RAY_BIN}" stop --force >/dev/null 2>&1 || true; }
if [[ "${MOPD_CONFIG_ONLY:-0}" != "1" ]]; then
  trap cleanup EXIT INT TERM
  cleanup
fi

for model in "${STUDENT_MODEL}" "${MATH_TEACHER}" "${CODE_TEACHER}" "${INSTRUCT_TEACHER}"; do
  test -f "${model}/config.json" || { echo "Missing model config: ${model}" >&2; exit 1; }
  compgen -G "${model}/*.safetensors" >/dev/null || { echo "Missing model weights: ${model}" >&2; exit 1; }
done
test -f "${PREFIX_DATASET}" || { echo "Missing Prefix-256 dataset: ${PREFIX_DATASET}" >&2; exit 1; }
ROWS=$("${PYTHON_BIN}" -c "import pyarrow.parquet as pq; print(pq.ParquetFile('${PREFIX_DATASET}').metadata.num_rows)")
test "${ROWS}" -eq 67836 || { echo "Unexpected Prefix-256 rows: ${ROWS}" >&2; exit 1; }

if [[ "${MOPD_CONFIG_ONLY:-0}" != "1" ]] && [[ -f "${MERGED_MODEL}/config.json" ]] && compgen -G "${MERGED_MODEL}/*.safetensors" >/dev/null; then
  echo "Same-origin Prefix-256 MOPD already merged: ${MERGED_MODEL}"
  exit 0
fi

if [[ "${MOPD_CONFIG_ONLY:-0}" == "1" || ! -d "${ACTOR_CHECKPOINT}" ]]; then
  "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct algorithm.use_kl_in_reward=False \
    data.train_files="${PREFIX_DATASET}" data.val_files="${PREFIX_DATASET}" data.train_batch_size=140 +data.gen_batch_size=140 data.shuffle=False data.dataloader_num_workers=0 \
    data.val_max_samples=20 data.val_batch_size=20 \
    data.sampler.class_path=pkg://verl.mopd.domain_sampler data.sampler.class_name=FixedRatioMOPDSampler \
    data.max_prompt_length=2304 +data.teacher_prefix_max_len=256 data.max_response_length=7168 \
    data.filter_overlong_prompts=True data.filter_overlong_prompts_workers=1 data.truncation=error data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +actor_rollout_ref.actor.teacher_prefix_sft_loss_coef=0.1 actor_rollout_ref.actor.teacher_prefix_sft_loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=40 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9472 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.checkpoint.save_contents='[model]' actor_rollout_ref.actor.checkpoint.load_contents='[model]' \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.rollout.log_prob_top_k=16 +actor_rollout_ref.rollout.top_k_strategy=only_stu \
    +actor_rollout_ref.rollout.reward_weight_mode=student_p +actor_rollout_ref.rollout.teacher_temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 actor_rollout_ref.rollout.max_model_len=9472 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=False reward_model.reward_manager=batch \
    trainer.n_gpus_per_node="${STUDENT_GPUS}" trainer.nnodes=1 trainer.total_epochs=1 trainer.balance_batch=True \
    trainer.val_before_train=False trainer.save_freq="${EXPECTED_STEP}" trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard"]' trainer.project_name=MOPD trainer.experiment_name="${RUN_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_ROOT}" trainer.validation_data_dir="validation_log/mopd/${RUN_NAME}" \
    +mopd.enable=True +mopd.student_gpus="${STUDENT_GPUS}" +mopd.advantage_clip=5 +mopd.prefix.enable=True \
    +mopd.domain_ratios.math=0.35 +mopd.domain_ratios.instruct=0.35 +mopd.domain_ratios.code=0.30 \
    +mopd.teachers.math.model_path="${MATH_TEACHER}" +mopd.teachers.instruct.model_path="${INSTRUCT_TEACHER}" \
    +mopd.teachers.code.model_path="${CODE_TEACHER}" +mopd.teachers.math.micro_batch_size=1 \
    +mopd.teachers.instruct.micro_batch_size=1 +mopd.teachers.code.micro_batch_size=1 \
    "${HYDRA_ARGS[@]}"
fi

[[ "${MOPD_CONFIG_ONLY:-0}" == "1" ]] && exit 0
cleanup
test -d "${ACTOR_CHECKPOINT}" || { echo "Missing final actor checkpoint: ${ACTOR_CHECKPOINT}" >&2; exit 1; }
test ! -e "${MERGED_MODEL}" || { echo "Incomplete merge target exists: ${MERGED_MODEL}" >&2; exit 1; }
MERGE_STAGE="${MERGED_MODEL}.tmp.$$"
"${PYTHON_BIN}" -m verl.model_merger merge --backend fsdp --local_dir "${ACTOR_CHECKPOINT}" --target_dir "${MERGE_STAGE}"
test -f "${MERGE_STAGE}/config.json" && compgen -G "${MERGE_STAGE}/*.safetensors" >/dev/null || { echo "Incomplete merged model" >&2; exit 1; }
mv "${MERGE_STAGE}" "${MERGED_MODEL}"
echo "Same-origin Prefix-256 MOPD complete: ${MERGED_MODEL}"
