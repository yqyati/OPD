"""EOS bridging helpers for teacher/student pairs with different terminal IDs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_eos_bridge(
    canonical_eos_token_id: int | None,
    source_eos_token_id: int | None,
) -> tuple[int, int] | None:
    """Validate an optional source-EOS to canonical-EOS mapping."""
    if canonical_eos_token_id is None and source_eos_token_id is None:
        return None
    if canonical_eos_token_id is None or source_eos_token_id is None:
        raise ValueError("canonical_eos_token_id and source_eos_token_id must be configured together.")
    return int(canonical_eos_token_id), int(source_eos_token_id)


def canonicalize_completed_prefix_terminal_eos(
    token_ids: Sequence[int],
    *,
    canonical_eos_token_id: int | None,
    source_eos_token_id: int | None,
) -> list[int]:
    """Map the terminal EOS of a completed teacher prefix into student vocabulary."""
    bridge = validate_eos_bridge(canonical_eos_token_id, source_eos_token_id)
    canonicalized = [int(token_id) for token_id in token_ids]
    if bridge is None:
        return canonicalized

    canonical_eos_token_id, source_eos_token_id = bridge
    if not canonicalized or canonicalized[-1] != source_eos_token_id:
        raise RuntimeError(
            "A stopped teacher prefix must end in its configured source EOS; "
            f"got {canonicalized[-1] if canonicalized else None}, expected {source_eos_token_id}."
        )
    canonicalized[-1] = canonical_eos_token_id
    return canonicalized


def canonicalize_teacher_eos_logits_(
    logits: torch.Tensor,
    prediction_mask: torch.Tensor | None,
    *,
    canonical_eos_token_id: int | None,
    source_eos_token_id: int | None,
) -> None:
    """Merge source-EOS probability into canonical EOS at valid response steps."""
    bridge = validate_eos_bridge(canonical_eos_token_id, source_eos_token_id)
    if prediction_mask is None or bridge is None:
        return

    canonical_eos_token_id, source_eos_token_id = bridge
    if canonical_eos_token_id == source_eos_token_id:
        return
    if logits.ndim != 2 or prediction_mask.ndim != 1:
        raise ValueError(f"Expected logits=(N,V) and prediction mask=(N,), got {logits.shape=} {prediction_mask.shape=}")
    if logits.shape[0] != prediction_mask.shape[0]:
        raise ValueError(f"Prediction-mask/logit row mismatch: {prediction_mask.shape=} {logits.shape=}")
    prediction_mask = prediction_mask.bool()
    if not prediction_mask.any():
        return
    if not (0 <= canonical_eos_token_id < logits.shape[-1] and 0 <= source_eos_token_id < logits.shape[-1]):
        raise ValueError(
            f"EOS ids must be in teacher vocabulary [0, {logits.shape[-1]}): "
            f"{canonical_eos_token_id=}, {source_eos_token_id=}"
        )

    valid_logits = logits[prediction_mask]
    valid_logits[:, canonical_eos_token_id] = torch.logaddexp(
        valid_logits[:, canonical_eos_token_id], valid_logits[:, source_eos_token_id]
    )
    valid_logits[:, source_eos_token_id] = -float("inf")
    logits[prediction_mask] = valid_logits
