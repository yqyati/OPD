"""Domain routing and order-preserving scatter helpers for MOPD teacher scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .config import DOMAIN_ORDER


@dataclass(frozen=True)
class RoutePlan:
    """The deterministic partition of one student-rollout batch by domain."""

    batch_size: int
    indices_by_domain: Mapping[str, torch.Tensor]

    def validate_complete(self) -> None:
        all_indices = torch.cat([self.indices_by_domain[domain].cpu() for domain in DOMAIN_ORDER])
        expected = torch.arange(self.batch_size, dtype=torch.long)
        if not torch.equal(torch.sort(all_indices).values, expected):
            raise ValueError("MOPD route plan must contain every batch index exactly once")


def build_route_plan(domains: Sequence[object]) -> RoutePlan:
    """Build a route plan from a batch-aligned array of domain labels."""
    normalized = [str(domain) for domain in domains]
    invalid = sorted(set(normalized).difference(DOMAIN_ORDER))
    if invalid:
        raise ValueError(f"Unknown MOPD domains in batch: {invalid}; expected only {DOMAIN_ORDER}")
    indices_by_domain = {
        domain: torch.tensor([index for index, value in enumerate(normalized) if value == domain], dtype=torch.long)
        for domain in DOMAIN_ORDER
    }
    plan = RoutePlan(batch_size=len(normalized), indices_by_domain=indices_by_domain)
    plan.validate_complete()
    return plan


def scatter_routed_tensors(
    plan: RoutePlan,
    tensors_by_domain: Mapping[str, torch.Tensor],
    *,
    field_name: str,
) -> torch.Tensor:
    """Restore domain-sub-batch tensors to the original student-batch order.

    Each tensor must have its batch dimension in the same order as the matching
    ``RoutePlan.indices_by_domain``.  Shape/dtype/device consistency is checked
    before allocation so a wrong teacher response fails loudly rather than being
    silently assigned to another domain's samples.
    """
    if set(tensors_by_domain) != set(DOMAIN_ORDER):
        raise ValueError(f"{field_name}: expected results for {DOMAIN_ORDER}, got {sorted(tensors_by_domain)}")
    reference = None
    for domain in DOMAIN_ORDER:
        tensor = tensors_by_domain[domain]
        expected_rows = len(plan.indices_by_domain[domain])
        if tensor.ndim < 1 or tensor.shape[0] != expected_rows:
            raise ValueError(
                f"{field_name}: {domain} returned shape {tuple(tensor.shape)}, expected first dimension {expected_rows}"
            )
        if reference is None:
            reference = tensor
        elif (
            tensor.shape[1:] != reference.shape[1:]
            or tensor.dtype != reference.dtype
            or tensor.device != reference.device
        ):
            raise ValueError(
                f"{field_name}: all teacher results must share trailing shape, dtype, and device; "
                f"got {domain}={tuple(tensor.shape)}/{tensor.dtype}/{tensor.device}, "
                f"reference={tuple(reference.shape)}/{reference.dtype}/{reference.device}"
            )
    assert reference is not None
    output = torch.empty((plan.batch_size, *reference.shape[1:]), dtype=reference.dtype, device=reference.device)
    for domain in DOMAIN_ORDER:
        output.index_copy_(0, plan.indices_by_domain[domain].to(reference.device), tensors_by_domain[domain])
    return output
