#!/usr/bin/env bash
# Plain MOPD initialized from the three-domain full-7168 SFT checkpoint.
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
SFT_MODEL_DIR=${SFT_MODEL_DIR:?SFT_MODEL_DIR must point to the merged full-7168 SFT model}
EXPECTED_STEP=484
CHECKPOINT_ROOT=checkpoint/mopd/q8b_pragmatic_full7168sftinit_plain_mopd_8gpu
ACTOR_CHECKPOINT="${CHECKPOINT_ROOT}/global_step_${EXPECTED_STEP}/actor"
MERGED_MODEL=merged_models/q8b_pragmatic_full7168sftinit_plain_mopd_8gpu_step${EXPECTED_STEP}
HYDRA_ARGS=()
if [ "${MOPD_CONFIG_ONLY:-0}" = "1" ]; then
  HYDRA_ARGS=(--cfg job)
fi

test -f "${SFT_MODEL_DIR}/config.json" || { echo "Missing full-SFT model config: ${SFT_MODEL_DIR}/config.json" >&2; exit 1; }
compgen -G "${SFT_MODEL_DIR}/*.safetensors" >/dev/null || { echo "Missing full-SFT model weights: ${SFT_MODEL_DIR}/*.safetensors" >&2; exit 1; }

cleanup() {
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
}
if [[ "${MOPD_CONFIG_ONLY:-0}" != "1" ]]; then
  trap cleanup EXIT INT TERM
  cleanup
fi

if [[ "${MOPD_CONFIG_ONLY:-0}" != "1" ]] && [[ -f "${MERGED_MODEL}/config.json" ]] && compgen -G "${MERGED_MODEL}/*.safetensors" >/dev/null; then
  echo "Full-SFT initialized plain MOPD is already complete: ${MERGED_MODEL}"
  exit 0
fi

if [[ "${MOPD_CONFIG_ONLY:-0}" == "1" || ! -d "${ACTOR_CHECKPOINT}" ]]; then
  "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=token_reward_direct \
  algorithm.use_kl_in_reward=False \
  data.train_files=datasets/mopd/q8b_pragmatic_plain_mopd_math_code_instruct_prompts.parquet \
  data.val_files=datasets/mopd/q8b_pragmatic_plain_mopd_math_code_instruct_prompts.parquet \
  data.train_batch_size=140 +data.gen_batch_size=140 data.shuffle=False data.dataloader_num_workers=0 \
  data.val_max_samples=20 data.val_batch_size=20 \
  data.sampler.class_path=pkg://verl.mopd.domain_sampler data.sampler.class_name=FixedRatioMOPDSampler \
  data.max_prompt_length=2048 data.max_response_length=4096 data.filter_overlong_prompts=True data.filter_overlong_prompts_workers=1 data.truncation=error data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path="${SFT_MODEL_DIR}" actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=140 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.checkpoint.save_contents='[model]' actor_rollout_ref.actor.checkpoint.load_contents='[model]' \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.repetition_penalty=1.0 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 +actor_rollout_ref.rollout.log_prob_top_k=16 \
  +actor_rollout_ref.rollout.top_k_strategy=only_stu +actor_rollout_ref.rollout.reward_weight_mode=student_p +actor_rollout_ref.rollout.teacher_temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
  actor_rollout_ref.rollout.max_model_len=6144 actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.enable=False reward_model.reward_manager=batch \
  trainer.n_gpus_per_node=5 trainer.nnodes=1 trainer.total_epochs=1 \
  trainer.balance_batch=True trainer.val_before_train=False trainer.save_freq=484 trainer.logger='["console","tensorboard"]' \
  trainer.project_name=MOPD trainer.experiment_name=q8b_pragmatic_full7168sftinit_plain_mopd_8gpu \
  trainer.default_local_dir=checkpoint/mopd/q8b_pragmatic_full7168sftinit_plain_mopd_8gpu \
  trainer.validation_data_dir=validation_log/mopd/q8b_pragmatic_full7168sftinit_plain_mopd_8gpu \
  +mopd.enable=True +mopd.student_gpus=5 +mopd.advantage_clip=5 \
  +mopd.domain_ratios.math=0.35 +mopd.domain_ratios.instruct=0.35 +mopd.domain_ratios.code=0.30 \
  +mopd.teachers.math.model_path="${OPD_ROOT}/merged_models/q8b_q30ba3b_dapo_math17k_think_correct6108_max7168_sftinit_math_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step373" \
  +mopd.teachers.instruct.model_path="${OPD_ROOT}/merged_models/q8b_q30ba3b_ifrlvr_think_correct5917_sftinit_ifrlvr_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step516" \
  +mopd.teachers.code.model_path="${OPD_ROOT}/merged_models/q8b_q30ba3b_eurus_code_think_correct_max7168_sftinit_code_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step523" \
  +mopd.teachers.math.micro_batch_size=1 +mopd.teachers.instruct.micro_batch_size=1 +mopd.teachers.code.micro_batch_size=1 \
  "${HYDRA_ARGS[@]}"
fi

[[ "${MOPD_CONFIG_ONLY:-0}" == "1" ]] && exit 0
cleanup
test -d "${ACTOR_CHECKPOINT}" || { echo "Training exited without the required final actor checkpoint: ${ACTOR_CHECKPOINT}" >&2; exit 1; }
test ! -e "${MERGED_MODEL}" || { echo "Incomplete merge target already exists; inspect it before retrying: ${MERGED_MODEL}" >&2; exit 1; }
MERGE_STAGE="${MERGED_MODEL}.tmp.$$"
"${PYTHON_BIN}" -m verl.model_merger merge --backend fsdp --local_dir "${ACTOR_CHECKPOINT}" --target_dir "${MERGE_STAGE}"
test -f "${MERGE_STAGE}/config.json" || { echo "Merged model has no config.json: ${MERGE_STAGE}" >&2; exit 1; }
compgen -G "${MERGE_STAGE}/*.safetensors" >/dev/null || { echo "Merged model has no safetensors: ${MERGE_STAGE}" >&2; exit 1; }
mv "${MERGE_STAGE}" "${MERGED_MODEL}"
echo "Full-SFT initialized plain MOPD checkpoint and merged model are complete: ${MERGED_MODEL}"
