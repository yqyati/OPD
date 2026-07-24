#!/usr/bin/env python3
"""Describe the textual stage reached by a teacher prefix budget.

This is a diagnosis only: it decodes the exact generated token IDs stored in a
teacher-prefix parquet and reports how much of the trace remains at a selected
budget. It never asks another model to judge the trace.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ANSWER_PATTERN = re.compile(
    r"(?:\\boxed\b|\\fbox\b|final\s+answer|the\s+answer\s+is|answer\s*[:=]|therefore\s*,?\s*the\s+answer)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def has_answer_marker(text: str) -> bool:
    return bool(ANSWER_PATTERN.search(text))


def quantile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[round((len(sorted_values) - 1) * fraction)]


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise ValueError("--budget must be positive")

    source = pq.ParquetFile(args.input)
    required = {"teacher_prefix_token_ids", "teacher_prefix_finish_reason"}
    missing = required.difference(source.schema_arrow.names)
    if missing:
        raise RuntimeError(f"missing required columns: {sorted(missing)}")

    rows = []
    for batch in source.iter_batches(batch_size=128, columns=sorted(required)):
        rows.extend(batch.to_pylist())

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    budget = args.budget
    available = [row for row in rows if len(row["teacher_prefix_token_ids"] or []) >= budget]
    stopped_by_budget = [
        row
        for row in rows
        if row["teacher_prefix_finish_reason"] == "stop" and len(row["teacher_prefix_token_ids"] or []) <= budget
    ]

    remaining = [len(row["teacher_prefix_token_ids"]) - budget for row in available]
    prefix_answer_markers = 0
    suffix_answer_markers = 0
    full_answer_markers = 0
    for row in available:
        token_ids = [int(token) for token in row["teacher_prefix_token_ids"]]
        early = tokenizer.decode(token_ids[:budget], skip_special_tokens=False)
        late = tokenizer.decode(token_ids[budget:], skip_special_tokens=False)
        full = early + late
        prefix_answer_markers += has_answer_marker(early)
        suffix_answer_markers += has_answer_marker(late)
        full_answer_markers += has_answer_marker(full)

    rng = random.Random(args.seed)
    sample_rows = rng.sample(available, min(args.samples, len(available)))
    samples = []
    for row in sample_rows:
        token_ids = [int(token) for token in row["teacher_prefix_token_ids"]]
        samples.append(
            {
                "trace_len": len(token_ids),
                "finish_reason": row["teacher_prefix_finish_reason"],
                "prefix_0_to_budget": tokenizer.decode(token_ids[:budget], skip_special_tokens=False),
                "next_128_tokens": tokenizer.decode(token_ids[budget : 2 * budget], skip_special_tokens=False),
                "tail_after_budget": tokenizer.decode(token_ids[budget:], skip_special_tokens=False),
            }
        )

    summary = {
        "input": str(Path(args.input).resolve()),
        "budget": budget,
        "rows": len(rows),
        "rows_available_at_budget": len(available),
        "teacher_stopped_by_budget": len(stopped_by_budget),
        "teacher_alive_at_budget": len(rows) - len(stopped_by_budget),
        "remaining_tokens_when_available": {
            "mean": sum(remaining) / len(remaining) if remaining else 0.0,
            "p10": quantile(sorted(remaining), 0.10),
            "p25": quantile(sorted(remaining), 0.25),
            "p50": quantile(sorted(remaining), 0.50),
            "p75": quantile(sorted(remaining), 0.75),
            "p90": quantile(sorted(remaining), 0.90),
        },
        "answer_marker_rate_among_available": {
            "in_prefix_0_to_budget": prefix_answer_markers / len(available) if available else 0.0,
            "in_tail_after_budget": suffix_answer_markers / len(available) if available else 0.0,
            "in_full_trace": full_answer_markers / len(available) if available else 0.0,
        },
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
