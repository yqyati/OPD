#!/usr/bin/env python3
"""Keep complete, all-constraint-correct IF-RLVR teacher trajectories."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_OPEN_INSTRUCT = ROOT / "third_party" / "open-instruct-ifrlvr"
if str(OFFICIAL_OPEN_INSTRUCT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_OPEN_INSTRUCT))

from open_instruct.IFEvalG import instructions_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    return parser.parse_args()


def visible_answer(completion: str) -> str:
    answer = completion.replace("<|assistant|>", "").strip()
    answer = answer.split("</think>")[-1]
    return answer.replace("<answer>", "").replace("</answer>", "").strip()


def load_constraint(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    constraint: Any = ground_truth
    if isinstance(constraint, str):
        constraint = ast.literal_eval(constraint)
    if isinstance(constraint, list):
        if len(constraint) != 1:
            raise ValueError(f"Expected singleton constraint list, got {len(constraint)}")
        constraint = constraint[0]
    if isinstance(constraint, str):
        constraint = json.loads(constraint)
    if not isinstance(constraint, dict):
        raise TypeError(f"Unexpected constraint type: {type(constraint)!r}")
    return constraint


def score_item(item: tuple[str, str, str]) -> tuple[bool, int, int, str | None]:
    response, ground_truth, finish_reason = item
    if finish_reason != "stop" or "</think>" not in response:
        return False, 0, 0, None
    try:
        constraint = load_constraint(ground_truth)
        instruction_ids = constraint["instruction_id"]
        kwargs_list = constraint["kwargs"]
        if len(instruction_ids) != len(kwargs_list):
            raise ValueError("instruction_id and kwargs lengths differ")
        answer = visible_answer(response)
        passed = 0
        for instruction_id, arguments in zip(instruction_ids, kwargs_list, strict=True):
            arguments = {} if arguments is None else {key: value for key, value in arguments.items() if value is not None}
            checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
            checker.build_description(**arguments)
            passed += int(bool(checker.check_following(answer)))
        return passed == len(instruction_ids), passed, len(instruction_ids), None
    except Exception as error:  # Keep expensive generation usable despite isolated malformed rows.
        return False, 0, 0, type(error).__name__


def main() -> None:
    args = parse_args()
    dataframe = pd.read_parquet(args.input)
    required = {"reward_model", "teacher_response_text", "teacher_response_finish_reason"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    items = [
        (row.teacher_response_text, row.reward_model["ground_truth"], row.teacher_response_finish_reason)
        for row in dataframe[["teacher_response_text", "reward_model", "teacher_response_finish_reason"]].itertuples(index=False)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(score_item, items, chunksize=32))

    keep = [result[0] for result in results]
    dataframe["teacher_response_constraints_passed"] = [result[1] for result in results]
    dataframe["teacher_response_constraints_total"] = [result[2] for result in results]
    dataframe["teacher_response_verifier_error"] = [result[3] for result in results]
    output = dataframe.loc[keep].reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)

    natural = sum(item[2] == "stop" for item in items)
    print(f"input rows: {len(dataframe)}")
    print(f"natural-stop rows: {natural}")
    print(f"all-constraint-correct rows: {len(output)}")
    print(f"correct ratio: {len(output) / max(len(dataframe), 1):.4%}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
