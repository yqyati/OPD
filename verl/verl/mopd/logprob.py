import torch


def gather_log_probs_by_token_chunks(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    extra_ids: torch.Tensor | None = None,
    token_chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gather selected log probabilities without materializing full-sequence log-softmax."""
    if token_chunk_size <= 0:
        raise ValueError(f"token_chunk_size must be positive, got {token_chunk_size}")
    if target_ids.shape != logits.shape[:-1]:
        raise ValueError(
            f"target_ids must match logits leading dimensions {tuple(logits.shape[:-1])}, "
            f"got {tuple(target_ids.shape)}"
        )
    if extra_ids is not None and extra_ids.shape[:-1] != logits.shape[:-1]:
        raise ValueError(
            f"extra_ids must match logits leading dimensions {tuple(logits.shape[:-1])}, "
            f"got {tuple(extra_ids.shape)}"
        )

    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_target_ids = target_ids.reshape(-1, 1)
    flat_extra_ids = None if extra_ids is None else extra_ids.reshape(flat_logits.shape[0], -1)

    target_parts = []
    extra_parts = []
    for start in range(0, flat_logits.shape[0], token_chunk_size):
        end = min(start + token_chunk_size, flat_logits.shape[0])
        chunk_log_probs = torch.log_softmax(flat_logits[start:end], dim=-1)
        target_parts.append(chunk_log_probs.gather(dim=-1, index=flat_target_ids[start:end]))
        if flat_extra_ids is not None:
            extra_parts.append(chunk_log_probs.gather(dim=-1, index=flat_extra_ids[start:end]))

    target_log_probs = torch.cat(target_parts, dim=0).reshape(target_ids.shape)
    if flat_extra_ids is None:
        return target_log_probs, None

    extra_log_probs = torch.cat(extra_parts, dim=0).reshape(extra_ids.shape)
    return target_log_probs, extra_log_probs


def entropy_from_logits_by_token_chunks(
    logits: torch.Tensor,
    token_chunk_size: int = 256,
) -> torch.Tensor:
    """Compute categorical entropy without materializing full-sequence probabilities."""
    if token_chunk_size <= 0:
        raise ValueError(f"token_chunk_size must be positive, got {token_chunk_size}")
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    entropy_parts = []
    for start in range(0, flat_logits.shape[0], token_chunk_size):
        end = min(start + token_chunk_size, flat_logits.shape[0])
        chunk_log_probs = torch.log_softmax(flat_logits[start:end], dim=-1)
        chunk_probs = chunk_log_probs.exp()
        entropy_parts.append(-(chunk_probs * chunk_log_probs).sum(dim=-1))
    return torch.cat(entropy_parts, dim=0).reshape(logits.shape[:-1])
