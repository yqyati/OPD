"""Validated configuration objects for the isolated MOPD runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DOMAIN_ORDER = ("math", "instruct", "code")


@dataclass(frozen=True)
class TeacherSpec:
    """One frozen teacher service and the checkpoint it owns."""

    domain: str
    model_path: str
    num_gpus: int = 1
    micro_batch_size: int = 1

    def validate(self, *, require_local_paths: bool = False) -> None:
        if self.domain not in DOMAIN_ORDER:
            raise ValueError(f"Unsupported MOPD domain {self.domain!r}; expected one of {DOMAIN_ORDER}")
        if not self.model_path:
            raise ValueError(f"MOPD teacher {self.domain!r} has an empty model_path")
        if self.num_gpus != 1:
            raise ValueError("The initial asynchronous MOPD service supports exactly one GPU per teacher")
        if self.micro_batch_size < 1:
            raise ValueError("teacher micro_batch_size must be positive")
        if require_local_paths and not (Path(self.model_path) / "config.json").is_file():
            raise FileNotFoundError(f"MOPD teacher checkpoint has no config.json: {self.model_path}")


@dataclass(frozen=True)
class MOPDConfig:
    """Runtime contract shared by plain-MOPD and Prefix-MOPD launchers.

    The first implementation intentionally supports the policy-gradient / sampled
    token path (`top_k_strategy=only_stu`).  Prefix-specific loss fields are kept
    outside this object so that the existing actor implementation remains the
    sole owner of prefix-loss math.
    """

    enabled: bool
    teachers: Mapping[str, TeacherSpec]
    domain_ratios: Mapping[str, float]
    advantage_clip: float = 5.0
    request_timeout_s: float = 900.0
    prefix_enabled: bool = False
    student_gpus: int = 5

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> MOPDConfig:
        teacher_raw = raw.get("teachers", {})
        if not isinstance(teacher_raw, Mapping):
            raise ValueError("mopd.teachers must be a mapping")
        teachers = {}
        for domain, value in teacher_raw.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"mopd.teachers.{domain} must be a mapping")
            teachers[str(domain)] = TeacherSpec(
                domain=str(domain),
                model_path=str(value.get("model_path", "")),
                num_gpus=int(value.get("num_gpus", 1)),
                micro_batch_size=int(value.get("micro_batch_size", 1)),
            )
        ratios = raw.get("domain_ratios", {})
        if not isinstance(ratios, Mapping):
            raise ValueError("mopd.domain_ratios must be a mapping")
        config = cls(
            enabled=bool(raw.get("enable", False)),
            teachers=teachers,
            domain_ratios={str(key): float(value) for key, value in ratios.items()},
            advantage_clip=float(raw.get("advantage_clip", 5.0)),
            request_timeout_s=float(raw.get("request_timeout_s", 900.0)),
            prefix_enabled=(
                bool(raw.get("prefix", {}).get("enable", False))
                if isinstance(raw.get("prefix", {}), Mapping)
                else False
            ),
            student_gpus=int(raw.get("student_gpus", 5)),
        )
        config.validate(require_local_paths=True)
        return config

    def validate(self, *, require_local_paths: bool = False) -> None:
        if not self.enabled:
            return
        if set(self.teachers) != set(DOMAIN_ORDER):
            raise ValueError(f"MOPD needs exactly teachers for {DOMAIN_ORDER}; got {sorted(self.teachers)}")
        if set(self.domain_ratios) != set(DOMAIN_ORDER):
            raise ValueError(f"MOPD needs ratios for {DOMAIN_ORDER}; got {sorted(self.domain_ratios)}")
        for domain in DOMAIN_ORDER:
            spec = self.teachers[domain]
            if spec.domain != domain:
                raise ValueError(f"Teacher mapping key {domain!r} disagrees with TeacherSpec.domain={spec.domain!r}")
            spec.validate(require_local_paths=require_local_paths)
        ratio_sum = sum(float(self.domain_ratios[domain]) for domain in DOMAIN_ORDER)
        if any(float(self.domain_ratios[domain]) <= 0 for domain in DOMAIN_ORDER):
            raise ValueError(f"MOPD domain ratios must all be positive: {dict(self.domain_ratios)}")
        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError(f"MOPD domain ratios must sum to 1.0, got {ratio_sum}")
        if self.advantage_clip <= 0:
            raise ValueError("MOPD advantage_clip must be positive")
        if self.request_timeout_s <= 0:
            raise ValueError("MOPD request_timeout_s must be positive")

    def validate_training_topology(self, ppo_config) -> None:
        """Reject launcher settings that would silently change the 5+3 MOPD plan."""
        self.validate()
        student_gpus = int(ppo_config.trainer.n_gpus_per_node) * int(ppo_config.trainer.nnodes)
        teacher_gpus = sum(spec.num_gpus for spec in self.teachers.values())
        if student_gpus != self.student_gpus or teacher_gpus != 3:
            raise ValueError(
                f"MOPD requires {self.student_gpus} student GPUs + 3 teacher GPUs, "
                f"got {student_gpus}+{teacher_gpus}"
            )
        if int(ppo_config.actor_rollout_ref.rollout.n) != 1:
            raise ValueError("MOPD requires actor_rollout_ref.rollout.n=1")
        if str(ppo_config.algorithm.adv_estimator) != "token_reward_direct":
            raise ValueError("MOPD requires algorithm.adv_estimator=token_reward_direct, matching OPD")
        if bool(ppo_config.algorithm.use_kl_in_reward):
            raise ValueError("MOPD requires algorithm.use_kl_in_reward=False, matching OPD")
        if bool(ppo_config.actor_rollout_ref.actor.use_kl_loss):
            raise ValueError("MOPD requires actor_rollout_ref.actor.use_kl_loss=False, matching OPD")
        if int(ppo_config.actor_rollout_ref.rollout.get("log_prob_top_k", 0)) <= 0:
            raise ValueError("MOPD requires rollout.log_prob_top_k > 0")
        if ppo_config.actor_rollout_ref.rollout.get("top_k_strategy", "only_stu") != "only_stu":
            raise ValueError("Initial MOPD supports only top_k_strategy=only_stu")
        if ppo_config.actor_rollout_ref.rollout.get("reward_weight_mode", "student_p") != "student_p":
            raise ValueError("MOPD requires rollout.reward_weight_mode=student_p, matching OPD")
        if float(ppo_config.actor_rollout_ref.rollout.temperature) != 1.0:
            raise ValueError("MOPD requires rollout.temperature=1.0, matching OPD")
        if float(ppo_config.actor_rollout_ref.rollout.get("teacher_temperature", 1.0)) != 1.0:
            raise ValueError("Initial MOPD requires rollout.teacher_temperature=1.0")
        if self.prefix_enabled:
            prefix_len = int(ppo_config.data.get("teacher_prefix_max_len", 0) or 0)
            prefix_sft_coef = float(
                ppo_config.actor_rollout_ref.actor.get("teacher_prefix_sft_loss_coef", 0.0) or 0.0
            )
            if prefix_len <= 0:
                raise ValueError("Prefix-MOPD requires data.teacher_prefix_max_len > 0")
            if prefix_sft_coef <= 0:
                raise ValueError("Prefix-MOPD requires actor.teacher_prefix_sft_loss_coef > 0")
