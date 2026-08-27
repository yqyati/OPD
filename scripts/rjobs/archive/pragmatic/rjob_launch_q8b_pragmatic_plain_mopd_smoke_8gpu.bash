#!/usr/bin/env bash
# Qwen3-8B pragmatic plain-MOPD smoke: 1 student GPU + 3 routed teacher services (4-GPU smoke).
# Conservative memory settings are smoke-only; formal 8-GPU MOPD uses a separate profile.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl
export YANGQINGYU_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu
export OPD_ROOT="${YANGQINGYU_ROOT}/workspace/OPD"
export PYTHONPATH="${OPD_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
cd "${OPD_ROOT}"
PYTHON_BIN=/root/miniconda3/envs/verl/bin/python
RAY_BIN=/root/miniconda3/envs/verl/bin/ray
cleanup() {
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM
cleanup
for p in \
  datasets/mopd/q8b_pragmatic_plain_mopd_math_code_instruct_prompts.parquet \
  merged_models/q8b_q30ba3b_dapo_math17k_think_correct6108_max7168_sftinit_math_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step373/config.json \
  merged_models/q8b_q30ba3b_ifrlvr_think_correct5917_sftinit_ifrlvr_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step516/config.json \
  merged_models/q8b_q30ba3b_eurus_code_think_correct_max7168_sftinit_code_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step523/config.json; do
  test -e "$p" || { echo "Missing MOPD prerequisite: $p" >&2; exit 1; }
done
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=token_reward_direct algorithm.use_kl_in_reward=False data.train_files=datasets/mopd/q8b_pragmatic_plain_mopd_math_code_instruct_prompts.parquet \
  data.val_files=datasets/mopd/q8b_pragmatic_plain_mopd_math_code_instruct_prompts.parquet \
  data.train_batch_size=20 +data.gen_batch_size=20 data.shuffle=False data.dataloader_num_workers=0 \
  data.val_max_samples=20 data.val_batch_size=20 \
  data.sampler.class_path=pkg://verl.mopd.domain_sampler data.sampler.class_name=FixedRatioMOPDSampler \
  data.max_prompt_length=8192 data.max_response_length=1024 data.filter_overlong_prompts=True data.filter_overlong_prompts_workers=1 data.truncation=error data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path="${YANGQINGYU_ROOT}/model/Qwen3-8B-Base" actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=20 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.repetition_penalty=1.0 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 +actor_rollout_ref.rollout.log_prob_top_k=16 +actor_rollout_ref.rollout.top_k_strategy=only_stu +actor_rollout_ref.rollout.reward_weight_mode=student_p +actor_rollout_ref.rollout.teacher_temperature=1.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.15 actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.max_model_len=9216 actor_rollout_ref.rollout.max_num_batched_tokens=9216 actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False reward_model.enable=False reward_model.reward_manager=batch \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_training_steps=2 trainer.total_epochs=1 trainer.val_before_train=False \
  trainer.project_name=MOPD trainer.experiment_name=q8b_pragmatic_plain_mopd_smoke \
  trainer.default_local_dir=checkpoint/mopd/q8b_pragmatic_plain_smoke trainer.validation_data_dir=validation_log/mopd/q8b_pragmatic_plain_smoke trainer.rollout_audit_dir=null trainer.logger='["console","tensorboard"]' \
  +mopd.enable=True +mopd.student_gpus=1 +mopd.advantage_clip=5 +mopd.domain_ratios.math=0.35 +mopd.domain_ratios.instruct=0.35 +mopd.domain_ratios.code=0.30 \
  +mopd.teachers.math.model_path="${OPD_ROOT}/merged_models/q8b_q30ba3b_dapo_math17k_think_correct6108_max7168_sftinit_math_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step373" \
  +mopd.teachers.instruct.model_path="${OPD_ROOT}/merged_models/q8b_q30ba3b_ifrlvr_think_correct5917_sftinit_ifrlvr_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step516" \
  +mopd.teachers.code.model_path="${OPD_ROOT}/merged_models/q8b_q30ba3b_eurus_code_think_correct_max7168_sftinit_code_grpo_r7168_n8_b48_ep1_shuffle42_lr5e-6_step523" \
  +mopd.teachers.math.micro_batch_size=1 +mopd.teachers.instruct.micro_batch_size=1 +mopd.teachers.code.micro_batch_size=1
