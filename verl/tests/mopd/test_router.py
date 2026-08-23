import pytest
import torch
from omegaconf import OmegaConf

from verl.mopd.config import MOPDConfig, TeacherSpec
from verl.mopd.objective import clipped_teacher_advantage, compute_mopd_only_student_reward
from verl.mopd.prefill_worker import PrefillPayload
from verl.mopd.router import build_route_plan, scatter_routed_tensors
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, compute_token_reward_direct_advantage


def test_route_and_scatter_preserves_original_batch_order():
    domains = ["code", "math", "instruct", "math", "code"]
    plan = build_route_plan(domains)
    tensors = {
        "math": torch.tensor([[10.0], [30.0]]),
        "instruct": torch.tensor([[20.0]]),
        "code": torch.tensor([[0.0], [40.0]]),
    }
    restored = scatter_routed_tensors(plan, tensors, field_name="teacher_on_student_log_probs")
    assert restored.tolist() == [[0.0], [10.0], [20.0], [30.0], [40.0]]


def test_route_rejects_unknown_domain():
    with pytest.raises(ValueError, match="Unknown MOPD domains"):
        build_route_plan(["math", "biology"])


def test_scatter_rejects_wrong_teacher_batch_shape():
    plan = build_route_plan(["math", "code", "instruct"])
    with pytest.raises(ValueError, match="expected first dimension"):
        scatter_routed_tensors(
            plan,
            {
                "math": torch.zeros(2, 3),
                "instruct": torch.zeros(1, 3),
                "code": torch.zeros(1, 3),
            },
            field_name="teacher_on_student_log_probs",
        )


def test_config_requires_exact_domain_registry():
    config = MOPDConfig(
        enabled=True,
        teachers={
            "math": TeacherSpec("math", "/tmp/math"),
            "code": TeacherSpec("code", "/tmp/code"),
        },
        domain_ratios={"math": 0.5, "code": 0.5},
    )
    with pytest.raises(ValueError, match="exactly teachers"):
        config.validate()


def test_paper_teacher_advantage_is_two_sided_clipped():
    teacher = torch.tensor([[7.0, -8.0, 0.5]])
    student = torch.tensor([[0.0, 0.0, 0.0]])
    advantage = clipped_teacher_advantage(teacher_log_probs=teacher, student_log_probs=student, max_abs_value=5.0)
    assert advantage.tolist() == [[5.0, -5.0, 0.5]]


def test_mopd_reward_masks_padding_before_sequence_reduction():
    student = torch.zeros(2, 4, 2)
    teacher = torch.tensor(
        [
            [[10.0, 10.0], [2.0, 2.0], [5.0, 5.0], [5.0, 5.0]],
            [[-10.0, -10.0], [-2.0, -2.0], [-1.0, -1.0], [5.0, 5.0]],
        ]
    )
    response_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    reward, diagnostics = compute_mopd_only_student_reward(
        student_log_probs=student,
        teacher_on_student_log_probs=teacher,
        response_mask=response_mask,
        reward_weight_mode="none",
        max_abs_advantage=5.0,
    )

    assert reward.shape == (2, 4, 2)
    assert reward.sum(dim=-1).tolist() == [[5.0, 2.0, 0.0, 0.0], [-5.0, -2.0, -1.0, 0.0]]
    assert torch.count_nonzero(reward * (1 - response_mask).unsqueeze(-1)) == 0
    assert torch.all(reward.abs().sum(dim=(-1, -2)) <= response_mask.sum(dim=-1) * 5.0)
    assert diagnostics["mopd_teacher_advantage_clipped_abs_max"].tolist() == [5.0, 5.0]


def test_mopd_reward_rejects_misaligned_response_mask():
    with pytest.raises(ValueError, match="response_mask must have shape"):
        compute_mopd_only_student_reward(
            student_log_probs=torch.zeros(2, 4, 2),
            teacher_on_student_log_probs=torch.zeros(2, 4, 2),
            response_mask=torch.ones(2, 3),
            reward_weight_mode="student_p",
            max_abs_advantage=5.0,
        )


def test_mopd_reward_advantage_and_policy_loss_preserve_topk_and_mask_padding():
    student = torch.tensor(
        [[[0.0, -1.0], [-0.5, -1.5], [-2.0, -3.0]]],
    )
    teacher = student + torch.tensor(
        [[[2.0, -2.0], [1.0, -1.0], [5.0, 5.0]]],
    )
    response_mask = torch.tensor([[1, 1, 0]])
    reward, _ = compute_mopd_only_student_reward(
        student_log_probs=student,
        teacher_on_student_log_probs=teacher,
        response_mask=response_mask,
        reward_weight_mode="student_p",
        max_abs_advantage=5.0,
    )
    advantages, returns = compute_token_reward_direct_advantage(reward, response_mask)

    assert reward.shape == advantages.shape == returns.shape == (1, 3, 2)
    assert torch.count_nonzero(advantages[:, 2]) == 0

    old_log_prob = student.detach().clone()
    log_prob = old_log_prob.clone().requires_grad_(True)
    actor_config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "clip_ratio_c": 3.0,
        }
    )
    loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=actor_config,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.count_nonzero(log_prob.grad[:, :2]) > 0
    assert torch.count_nonzero(log_prob.grad[:, 2]) == 0


def test_mopd_training_contract_matches_opd_estimator():
    config = MOPDConfig(
        enabled=True,
        teachers={domain: TeacherSpec(domain, f"/tmp/{domain}") for domain in ("math", "instruct", "code")},
        domain_ratios={"math": 0.35, "instruct": 0.35, "code": 0.30},
        student_gpus=5,
    )
    ppo_config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 5, "nnodes": 1},
            "algorithm": {"adv_estimator": "token_reward_direct", "use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": False},
                "rollout": {
                    "n": 1,
                    "temperature": 1.0,
                    "teacher_temperature": 1.0,
                    "log_prob_top_k": 16,
                    "top_k_strategy": "only_stu",
                    "reward_weight_mode": "student_p",
                },
            },
        }
    )

    config.validate_training_topology(ppo_config)
    ppo_config.algorithm.adv_estimator = "grpo"
    with pytest.raises(ValueError, match="adv_estimator=token_reward_direct"):
        config.validate_training_topology(ppo_config)

    ppo_config.algorithm.adv_estimator = "token_reward_direct"
    ppo_config.actor_rollout_ref.rollout.teacher_temperature = 0.7
    with pytest.raises(ValueError, match="teacher_temperature=1.0"):
        config.validate_training_topology(ppo_config)


def test_prefix_mopd_requires_cached_prefix_training_fields():
    config = MOPDConfig(
        enabled=True,
        teachers={domain: TeacherSpec(domain, f"/tmp/{domain}") for domain in ("math", "instruct", "code")},
        domain_ratios={"math": 0.35, "instruct": 0.35, "code": 0.30},
        prefix_enabled=True,
        student_gpus=5,
    )
    ppo_config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 5, "nnodes": 1},
            "data": {"teacher_prefix_max_len": 0},
            "algorithm": {"adv_estimator": "token_reward_direct", "use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": False, "teacher_prefix_sft_loss_coef": 0.0},
                "rollout": {
                    "n": 1,
                    "temperature": 1.0,
                    "teacher_temperature": 1.0,
                    "log_prob_top_k": 16,
                    "top_k_strategy": "only_stu",
                    "reward_weight_mode": "student_p",
                },
            },
        }
    )

    with pytest.raises(ValueError, match="teacher_prefix_max_len"):
        config.validate_training_topology(ppo_config)
    ppo_config.data.teacher_prefix_max_len = 128
    with pytest.raises(ValueError, match="teacher_prefix_sft_loss_coef"):
        config.validate_training_topology(ppo_config)
    ppo_config.actor_rollout_ref.actor.teacher_prefix_sft_loss_coef = 0.1
    config.validate_training_topology(ppo_config)


def test_prefill_payload_requires_batch_aligned_topk_shape():
    payload = PrefillPayload.from_mapping({
        "input_ids": torch.zeros(2, 10, dtype=torch.long),
        "attention_mask": torch.ones(2, 10, dtype=torch.long),
        "position_ids": torch.zeros(2, 10, dtype=torch.long),
        "responses": torch.zeros(2, 4, dtype=torch.long),
        "student_top_k_ids": torch.zeros(2, 4, 16, dtype=torch.long),
    })
    assert payload.student_top_k_ids.shape == (2, 4, 16)
