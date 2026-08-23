"""MOPD-specific objective helpers, kept separate from legacy OPD math."""

from __future__ import annotations

import torch


def clipped_teacher_advantage(
    *, teacher_log_probs: torch.Tensor, student_log_probs: torch.Tensor, max_abs_value: float
) -> torch.Tensor:
    """Paper MOPD policy-gradient advantage: clip(log pi_T - log pi_S)."""
    if max_abs_value <= 0:
        raise ValueError("MOPD advantage clip must be positive")
    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError(
            f"Teacher/student log-prob shapes differ: {tuple(teacher_log_probs.shape)} vs "
            f"{tuple(student_log_probs.shape)}"
        )
    return torch.clamp(teacher_log_probs - student_log_probs, min=-max_abs_value, max=max_abs_value)


def compute_reward_weights(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Return candidate weights for the isolated MOPD objective."""
    if mode == "student_p":
        log_probs = student_log_probs
    elif mode == "teacher_p":
        log_probs = teacher_log_probs
    elif mode == "none":
        log_probs = torch.zeros_like(student_log_probs)
    else:
        raise ValueError(f"Unknown reward_weight_mode: {mode}")

    log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float("inf")))
    if normalize:
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    return torch.nan_to_num(torch.exp(log_probs), nan=0.0, posinf=0.0, neginf=0.0)


def compute_mopd_only_student_reward(
    *,
    student_log_probs: torch.Tensor,
    teacher_on_student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    reward_weight_mode: str,
    max_abs_advantage: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the MOPD only-student candidate reward.

    Inputs and returned rewards are ``[batch, response_length, top_k]`` to
    preserve legacy OPD's candidate-level policy-gradient contract.
    """
    if student_log_probs.shape != teacher_on_student_log_probs.shape or student_log_probs.ndim != 3:
        raise ValueError(
            "MOPD student/teacher candidate tensors must share shape "
            f"(batch, response_length, top_k); got {tuple(student_log_probs.shape)} and "
            f"{tuple(teacher_on_student_log_probs.shape)}"
        )
    if response_mask.shape != student_log_probs.shape[:2]:
        raise ValueError(
            "MOPD response_mask must have shape (batch, response_length); "
            f"got {tuple(response_mask.shape)} for log-probs {tuple(student_log_probs.shape)}"
        )
    response_mask = response_mask.to(device=student_log_probs.device, dtype=student_log_probs.dtype)
    valid_mask = torch.ones_like(student_log_probs, dtype=torch.bool)
    weights = compute_reward_weights(
        student_log_probs,
        teacher_on_student_log_probs,
        valid_mask,
        reward_weight_mode,
    )
    advantage = clipped_teacher_advantage(
        teacher_log_probs=teacher_on_student_log_probs,
        student_log_probs=student_log_probs,
        max_abs_value=float(max_abs_advantage),
    )
    expanded_response_mask = response_mask.unsqueeze(-1)
    candidate_reward = advantage * weights * expanded_response_mask
    batch_size = candidate_reward.shape[0]
    masked_unclipped_advantage = (
        teacher_on_student_log_probs - student_log_probs
    ).abs() * expanded_response_mask
    masked_clipped_advantage = advantage.abs() * expanded_response_mask
    diagnostics = {
        "mopd_teacher_advantage_unclipped_abs_max": masked_unclipped_advantage.amax().expand(batch_size),
        "mopd_teacher_advantage_clipped_abs_max": masked_clipped_advantage.amax().expand(batch_size),
    }
    return candidate_reward, diagnostics
