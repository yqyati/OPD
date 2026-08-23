"""Asynchronous Ray teacher-prefill service registry for MOPD."""

from __future__ import annotations

from typing import Mapping

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .config import DOMAIN_ORDER, TeacherSpec
from .prefill_worker import FrozenTeacherPrefill


@ray.remote
class TeacherPrefillService:
    """A GPU-isolated Ray actor that owns exactly one frozen teacher."""

    def __init__(self, spec: TeacherSpec, dtype: str = "bfloat16") -> None:
        spec.validate(require_local_paths=True)
        self.domain = spec.domain
        self.prefill = FrozenTeacherPrefill(
            model_path=spec.model_path,
            micro_batch_size=spec.micro_batch_size,
            dtype=dtype,
        )
        self.prefill.init_model()

    def score(self, payload):
        return self.prefill.score(payload)

    def health(self) -> dict[str, str]:
        return {"domain": self.domain, "status": "ready"}


def create_teacher_services(
    teachers: Mapping[str, TeacherSpec], *, dtype: str = "bfloat16", placement_groups=None
) -> dict[str, ray.actor.ActorHandle]:
    """Start one explicitly GPU-reserved service per MOPD domain teacher."""
    if set(teachers) != set(DOMAIN_ORDER):
        raise ValueError(f"Expected teacher specs for {DOMAIN_ORDER}, got {sorted(teachers)}")
    services = {}
    for domain in DOMAIN_ORDER:
        spec = teachers[domain]
        spec.validate(require_local_paths=True)
        options = {"num_gpus": spec.num_gpus}
        if placement_groups is not None:
            # The first (and only) MOPD teacher pool has three one-GPU bundles;
            # bind Math/IF/Code deterministically to 0/1/2.
            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=placement_groups[0], placement_group_bundle_index=DOMAIN_ORDER.index(domain)
            )
        services[domain] = TeacherPrefillService.options(**options).remote(spec, dtype=dtype)
    return services
