"""Multi-teacher on-policy distillation (MOPD).

This package is intentionally isolated from the existing single-teacher PPO / OPD
implementation.  The existing trainer calls this package only when explicitly
enabled by a MOPD launcher.
"""

from .config import MOPDConfig, TeacherSpec
from .router import DOMAIN_ORDER, RoutePlan, build_route_plan, scatter_routed_tensors

__all__ = [
    "DOMAIN_ORDER",
    "MOPDConfig",
    "RoutePlan",
    "TeacherSpec",
    "build_route_plan",
    "scatter_routed_tensors",
]
