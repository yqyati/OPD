"""Official AllenAI IF-RLVR reward for verl GRPO.

The supplied label is an IFEvalG constraint specification.  Each rollout is
scored by the fraction of its constraints satisfied after removing its Qwen
thinking section, matching AllenAI's IFEvalVerifier semantics.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_OPEN_INSTRUCT = ROOT / "third_party" / "open-instruct-ifrlvr"
if str(OFFICIAL_OPEN_INSTRUCT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_OPEN_INSTRUCT))

from open_instruct.IFEvalG import instructions_registry  # noqa: E402


def _visible_answer(completion: str) -> str:
    """Match AllenAI's removal of Qwen/OpenAI-style thinking and answer tags."""
    answer = completion.replace("<|assistant|>", "").strip()
    answer = answer.split("</think>")[-1]
    return answer.replace("<answer>", "").replace("</answer>", "").strip()


def _load_constraint(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    constraint: Any = ground_truth
    if isinstance(constraint, str):
        constraint = ast.literal_eval(constraint)
    if isinstance(constraint, list):
        if len(constraint) != 1:
            raise ValueError(f"Expected a singleton IF-RLVR constraint list, got {len(constraint)}")
        constraint = constraint[0]
    if isinstance(constraint, str):
        constraint = json.loads(constraint)
    if not isinstance(constraint, dict):
        raise TypeError(f"Unexpected IF-RLVR constraint: {type(constraint)!r}")
    return constraint


def reward_func(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **_kwargs) -> dict[str, Any]:
    """Return the mean satisfaction rate across a sample's constraints."""
    del data_source, extra_info
    answer = _visible_answer(solution_str)
    if not answer:
        return {"score": 0.0, "constraint_pass_rate": 0.0, "constraints_total": 0, "constraints_passed": 0}
    try:
        constraint = _load_constraint(ground_truth)
        instruction_ids = constraint["instruction_id"]
        kwargs_list = constraint["kwargs"]
        if len(instruction_ids) != len(kwargs_list):
            raise ValueError("instruction_id and kwargs lengths differ")
        passed = 0
        for instruction_id, arguments in zip(instruction_ids, kwargs_list, strict=True):
            arguments = {} if arguments is None else {key: value for key, value in arguments.items() if value is not None}
            checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
            checker.build_description(**arguments)
            passed += int(bool(checker.check_following(answer)))
        total = len(instruction_ids)
        score = passed / max(total, 1)
        return {
            "score": float(score),
            "constraint_pass_rate": float(score),
            "constraints_total": total,
            "constraints_passed": passed,
        }
    except Exception as error:
        # A malformed individual example must not terminate an expensive GRPO
        # job; preserve the error class in logs through the reward extra info.
        return {
            "score": 0.0,
            "constraint_pass_rate": 0.0,
            "constraints_total": 0,
            "constraints_passed": 0,
            "verifier_error": type(error).__name__,
        }
