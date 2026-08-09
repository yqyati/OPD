"""Binary answer verifier for the selected OpenScienceReasoning-2 science data."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any


BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", re.DOTALL)
LETTER_RE = re.compile(r"(?<![A-Z])([A-J])(?![A-Z])", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _final_segment(text: str) -> str:
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    answer_markers = list(re.finditer(r"(?:final\s+answer|answer)\s*(?:is|:)?", text, re.IGNORECASE))
    if answer_markers:
        return text[answer_markers[-1].end() :]
    return text.strip()


def _extract_letter(text: str) -> str | None:
    segment = _final_segment(text)
    matches = LETTER_RE.findall(segment)
    if matches:
        return matches[-1].upper()
    return None


def _extract_number(text: str) -> str | None:
    segment = _final_segment(text).replace(",", "")
    matches = NUMBER_RE.findall(segment)
    if matches:
        return matches[-1]
    return None


def _numeric_equal(prediction: str, target: str) -> bool:
    try:
        pred = Decimal(prediction)
        gold = Decimal(target)
    except (InvalidOperation, ValueError):
        return False
    if not pred.is_finite() or not gold.is_finite():
        return False
    return math.isclose(float(pred), float(gold), rel_tol=1e-4, abs_tol=1e-6)


def compute_score(solution_str: str, ground_truth: str, answer_type: str) -> dict[str, Any]:
    if answer_type == "mcq":
        prediction = _extract_letter(solution_str)
        correct = prediction == str(ground_truth).strip().upper()
    elif answer_type == "numeric":
        prediction = _extract_number(solution_str)
        correct = prediction is not None and _numeric_equal(prediction, str(ground_truth).strip())
    else:
        prediction = None
        correct = False
    return {
        "score": float(correct),
        "acc": bool(correct),
        "pred": prediction or "",
        "answer_type": answer_type,
    }


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **_kwargs):
    if data_source != "openscience_reasoning2_science":
        return {"score": 0.0, "acc": False, "pred": "", "answer_type": "unknown"}
    extra_info = extra_info or {}
    return compute_score(solution_str, str(ground_truth), str(extra_info.get("answer_type", "")))
