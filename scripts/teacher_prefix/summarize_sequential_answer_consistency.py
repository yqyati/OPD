#!/usr/bin/env python3
"""Evaluate a teacher-prefix selector using only student answer consistency."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

def read_records(output_dir: Path) -> tuple[dict[int, dict[int, list[dict[str, Any]]]], int]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    malformed = 0
    for path in sorted(output_dir.glob("handoff_rollouts_worker_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            grouped[int(row["source_row"])][int(row["requested_prefix_len"])].append(row)
    return grouped, malformed


def normalized_answer(response: str) -> str | None:
    start = response.rfind("\\boxed{")
    if start < 0:
        return None
    cursor = start + len("\\boxed{")
    depth = 1
    for index in range(cursor, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                answer = response[cursor:index]
                # Conservative formatting normalization only.  This never uses
                # the gold answer or symbolic equivalence.
                answer = (
                    answer.strip()
                    .replace(" ", "")
                    .replace("\n", "")
                    .replace("\\left", "")
                    .replace("\\right", "")
                    .replace("\\dfrac", "\\frac")
                    .replace("\\tfrac", "\\frac")
                )
                return answer or None
    return None


def state_consensus(records: list[dict[str, Any]]) -> tuple[float, str | None]:
    answers = [normalized_answer(str(record["response"])) for record in records]
    valid_answers = [answer for answer in answers if answer is not None]
    if not valid_answers:
        return 0.0, None
    modal_answer, modal_count = Counter(valid_answers).most_common(1)[0]
    return modal_count / len(records), modal_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--prefix-lengths", required=True)
    parser.add_argument("--min-consensus", type=float, default=0.5)
    parser.add_argument("--continuations-per-prefix", type=int, default=4)
    args = parser.parse_args()

    lengths = [int(value) for value in args.prefix_lengths.split(",") if value.strip()]
    if lengths != sorted(set(lengths)):
        raise ValueError("--prefix-lengths must be strictly increasing without duplicates")
    if not 0.0 < args.min_consensus <= 1.0:
        raise ValueError("--min-consensus must be in (0, 1]")

    output_dir = Path(args.rollout_dir)
    grouped, malformed = read_records(output_dir)
    states: list[dict[str, Any]] = []
    incomplete = 0
    for source_row, by_length in grouped.items():
        if any(len(by_length.get(length, [])) != args.continuations_per_prefix for length in lengths):
            incomplete += 1
            continue
        prompt_states = []
        for length in lengths:
            records = by_length[length]
            consensus, modal_answer = state_consensus(records)
            prompt_states.append(
                {
                    "source_row": source_row,
                    "prefix_len": length,
                    "answer_consensus": consensus,
                    "modal_answer": modal_answer,
                    "mean_rule_reward": sum(float(record["rule_score"]) for record in records) / len(records),
                }
            )
        states.extend(prompt_states)

    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        by_prompt[int(state["source_row"])].append(state)
    selected = []
    longest = lengths[-1]
    for prompt_states in by_prompt.values():
        prompt_states.sort(key=lambda state: int(state["prefix_len"]))
        choice = next((state for state in prompt_states if state["answer_consensus"] >= args.min_consensus), None)
        selected.append(
            {
                "source_row": prompt_states[0]["source_row"],
                "selected_prefix_len": int(choice["prefix_len"]) if choice is not None else longest,
                "fallback_to_longest": choice is None,
            }
        )

    bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        bins[f"{state['answer_consensus']:.2f}"].append(state)
    selected_counts = {str(length): sum(item["selected_prefix_len"] == length for item in selected) for length in lengths}
    summary = {
        "selector": "shortest_prefix_with_modal_student_answer_consensus_else_longest",
        "prefix_lengths": lengths,
        "min_consensus": args.min_consensus,
        "continuations_per_prefix": args.continuations_per_prefix,
        "complete_prompts": len(selected),
        "incomplete_prompts_excluded": incomplete,
        "malformed_rollout_records_skipped": malformed,
        "selected_length_counts": selected_counts,
        "mean_selected_prefix_len": sum(item["selected_prefix_len"] for item in selected) / len(selected),
        "fallback_to_longest_count": sum(item["fallback_to_longest"] for item in selected),
        "state_quality_by_answer_consensus": {
            label: {
                "num_states": len(bucket),
                "mean_rule_reward": sum(state["mean_rule_reward"] for state in bucket) / len(bucket),
            }
            for label, bucket in sorted(bins.items(), key=lambda item: float(item[0]))
        },
    }
    (output_dir / "answer_consistency_selector_records.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in selected)
    )
    (output_dir / "answer_consistency_selector_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
