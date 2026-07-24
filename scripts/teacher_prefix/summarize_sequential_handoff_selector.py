#!/usr/bin/env python3
"""Summarize a rollout-based shortest-sufficient teacher-prefix selector."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_worker_records(output_dir: Path) -> tuple[dict[int, dict[int, list[dict[str, Any]]]], int]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    paths = sorted(output_dir.glob("handoff_rollouts_worker_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no worker rollout shards found in {output_dir}")
    malformed_records = 0
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                # A vLLM worker can rarely leave an invalid JSON record behind.
                # The affected prompt is excluded below unless every candidate
                # still has the requested number of complete continuations.
                malformed_records += 1
                print(f"warning: skipping malformed rollout record: {path}:{line_number}: {error}", flush=True)
                continue
            grouped[int(row["source_row"])][int(row["requested_prefix_len"])].append(row)
    return grouped, malformed_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--prefix-lengths", required=True)
    parser.add_argument("--required-successes", type=int, default=2)
    parser.add_argument("--continuations-per-prefix", type=int, default=4)
    args = parser.parse_args()

    lengths = [int(value) for value in args.prefix_lengths.split(",") if value.strip()]
    if lengths != sorted(set(lengths)):
        raise ValueError("--prefix-lengths must be strictly increasing without duplicates")
    if args.required_successes < 1 or args.required_successes > args.continuations_per_prefix:
        raise ValueError("--required-successes must be in [1, continuations-per-prefix]")

    output_dir = Path(args.rollout_dir)
    grouped, malformed_records = read_worker_records(output_dir)
    selected: list[dict[str, Any]] = []
    incomplete: list[int] = []
    longest = lengths[-1]
    for source_row, by_length in grouped.items():
        if any(len(by_length.get(length, [])) != args.continuations_per_prefix for length in lengths):
            incomplete.append(source_row)
            continue
        successes = {
            length: sum(int(row["rule_score"]) for row in by_length[length])
            for length in lengths
        }
        selected_length = next((length for length in lengths if successes[length] >= args.required_successes), longest)
        selected.append(
            {
                "source_row": source_row,
                "selected_prefix_len": selected_length,
                "fallback_to_longest": selected_length == longest and successes[longest] < args.required_successes,
                "successes_by_prefix_len": {str(length): successes[length] for length in lengths},
            }
        )

    if not selected:
        raise RuntimeError("no complete prompt records available for selection")
    counts = {str(length): sum(item["selected_prefix_len"] == length for item in selected) for length in lengths}
    fallback_count = sum(item["fallback_to_longest"] for item in selected)
    summary = {
        "selector": "shortest_prefix_with_required_rollout_successes_else_longest",
        "prefix_lengths": lengths,
        "required_successes": args.required_successes,
        "continuations_per_prefix": args.continuations_per_prefix,
        "complete_prompts": len(selected),
        "incomplete_prompts_excluded": len(incomplete),
        "malformed_rollout_records_skipped": malformed_records,
        "selected_length_counts": counts,
        "mean_selected_prefix_len": sum(item["selected_prefix_len"] for item in selected) / len(selected),
        "fallback_to_longest_count": fallback_count,
    }
    (output_dir / "sequential_selector_records.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in selected)
    )
    (output_dir / "sequential_selector_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
