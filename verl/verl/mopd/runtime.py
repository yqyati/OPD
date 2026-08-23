"""Non-blocking submit/collect API used by the existing PPO loop when MOPD is enabled."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import ray
import torch

from verl.protocol import DataProto

from .config import DOMAIN_ORDER, MOPDConfig
from .router import RoutePlan, build_route_plan, scatter_routed_tensors
from .service import create_teacher_services


@dataclass
class PrefillTicket:
    plan: RoutePlan
    futures: Mapping[str, ray.ObjectRef]


def _payload_for_indices(batch: DataProto, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    required = ("input_ids", "attention_mask", "position_ids", "responses", "student_top_k_ids")
    missing = [name for name in required if name not in batch.batch.keys()]
    if missing:
        raise ValueError(f"MOPD needs student rollout fields not present in batch: {missing}")
    selected = batch.select_idxs(indices.tolist())
    return {name: selected.batch[name].detach().cpu() for name in required}


class AsyncMOPDRuntime:
    """Owns the three asynchronous teacher services for one MOPD run."""

    def __init__(self, config: MOPDConfig, *, dtype: str = "bfloat16", placement_groups=None) -> None:
        config.validate(require_local_paths=True)
        self.config = config
        self.placement_groups = list(placement_groups or [])
        self.services = {}
        try:
            self.services = create_teacher_services(config.teachers, dtype=dtype, placement_groups=placement_groups)
        except BaseException:
            # Actor construction can fail while loading one teacher. Clean up
            # actors already created in domain order and release the placement
            # group before propagating the original error.
            for service in self.services.values():
                ray.kill(service, no_restart=True)
            for placement_group in self.placement_groups:
                ray.util.remove_placement_group(placement_group)
            self.placement_groups.clear()
            raise

    def healthcheck(self) -> dict[str, dict[str, str]]:
        futures = [self.services[domain].health.remote() for domain in DOMAIN_ORDER]
        results = ray.get(futures, timeout=self.config.request_timeout_s)
        return {domain: result for domain, result in zip(DOMAIN_ORDER, results, strict=True)}

    def submit_prefill(self, batch: DataProto) -> PrefillTicket:
        if "mopd_domain" not in batch.non_tensor_batch:
            raise ValueError("MOPD batch has no non-tensor mopd_domain field")
        domains = np.asarray(batch.non_tensor_batch["mopd_domain"], dtype=object).tolist()
        plan = build_route_plan(domains)
        futures = {}
        for domain in DOMAIN_ORDER:
            indices = plan.indices_by_domain[domain]
            if len(indices) == 0:
                raise ValueError(f"MOPD batch contains no {domain} samples; use the fixed-ratio sampler")
            futures[domain] = self.services[domain].score.remote(_payload_for_indices(batch, indices))
        return PrefillTicket(plan=plan, futures=futures)

    def collect_prefill(self, ticket: PrefillTicket) -> DataProto:
        responses = ray.get([ticket.futures[domain] for domain in DOMAIN_ORDER], timeout=self.config.request_timeout_s)
        by_domain = {
            domain: response["teacher_on_student_log_probs"]
            for domain, response in zip(DOMAIN_ORDER, responses, strict=True)
        }
        teacher_log_probs = scatter_routed_tensors(
            ticket.plan, by_domain, field_name="teacher_on_student_log_probs"
        )
        return DataProto.from_dict(tensors={"teacher_on_student_log_probs": teacher_log_probs})

    def close(self) -> None:
        for service in self.services.values():
            ray.kill(service, no_restart=True)
        for placement_group in self.placement_groups:
            ray.util.remove_placement_group(placement_group)
        self.placement_groups.clear()
