"""No-op rule reward for token-only OPD runs.

The actor update uses the token-level distillation reward already stored in
``rm_scores``.  This avoids spending CPU time running code unit tests merely
to log a per-step true-reward metric.
"""


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return 0.0
