#!/usr/bin/env python3
"""Evaluate pre-specified, parameter-free combinations of state ranking scores."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ranks(values: list[float]) -> list[float]:
    """Average ranks, increasing with better scores, normalized within a prompt."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in range(start, end):
            output[order[index]] = rank
        start = end
    return output


def score_candidates(rows: list[dict[str, Any]]) -> None:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_row"]].append(row)
    for group in grouped.values():
        teacher = [row["teacher_token_support"] for row in group]
        student = [row["student_token_support"] for row in group]
        interaction = [row["trajectory_interaction_mass"] for row in group]
        handoff = [row["handoff_interaction_mass"] for row in group]
        entropy = [row["trajectory_overlap_entropy"] for row in group]
        rt, rs, ri, rh, re = map(ranks, (teacher, student, interaction, handoff, entropy))
        for index, row in enumerate(group):
            row["joint_support_mean"] = (teacher[index] + student[index]) / 2
            row["joint_support_min"] = min(teacher[index], student[index])
            row["support_agreement"] = -abs(teacher[index] - student[index])
            # Rank sums keep each signal comparable within exactly the four candidates.
            row["rank_interaction_student_support"] = ri[index] + rs[index]
            row["rank_interaction_teacher_support"] = ri[index] + rt[index]
            row["rank_interaction_joint_support"] = ri[index] + ranks([(t + s) / 2 for t, s in zip(teacher, student)])[index]
            row["rank_interaction_both_support"] = ri[index] + rt[index] + rs[index]
            row["rank_interaction_handoff_mass"] = ri[index] + rh[index]
            row["rank_interaction_entropy"] = ri[index] + re[index]


def evaluate(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, dict[str, float]]:
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["source_row"]].append(row)
    output = {}
    for metric in metrics:
        selected, random_values, pairs = [], [], []
        nonzero_selected, nonzero_random = [], []
        for group in by_prompt.values():
            scores = [row[metric] for row in group]
            values = [row["V_student"] for row in group]
            top = max(scores)
            selected_value = sum(value for score, value in zip(scores, values) if score == top) / sum(score == top for score in scores)
            selected.append(selected_value)
            random_values.append(sum(values) / len(values))
            if max(values) > min(values):
                nonzero_selected.append(selected_value)
                nonzero_random.append(sum(values) / len(values))
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    if values[left] == values[right]:
                        continue
                    direction = (scores[left] > scores[right]) - (scores[left] < scores[right])
                    truth = (values[left] > values[right]) - (values[left] < values[right])
                    pairs.append(1.0 if direction == truth else 0.5 if direction == 0 else 0.0)
        output[metric] = {
            "selected_V": sum(selected) / len(selected),
            "uplift": (sum(selected) - sum(random_values)) / len(selected),
            "pairwise_accuracy": sum(pairs) / len(pairs),
            "nonzero_gap_uplift": (sum(nonzero_selected) - sum(nonzero_random)) / len(nonzero_selected),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = load_rows(Path(args.scores))
    score_candidates(rows)
    candidates = [
        "trajectory_interaction_mass",
        "student_token_support",
        "teacher_token_support",
        "joint_support_mean",
        "joint_support_min",
        "support_agreement",
        "rank_interaction_student_support",
        "rank_interaction_teacher_support",
        "rank_interaction_joint_support",
        "rank_interaction_both_support",
        "rank_interaction_handoff_mass",
        "rank_interaction_entropy",
    ]
    result = {
        "all": evaluate(rows, candidates),
        "source_row_even": evaluate([row for row in rows if row["source_row"] % 2 == 0], candidates),
        "source_row_odd": evaluate([row for row in rows if row["source_row"] % 2 == 1], candidates),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
