#!/usr/bin/env python3
"""Create the paper's LiveCodeBench v6 February-May 2025 evaluation slice."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/livecodebench/v6/test6.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/livecodebench/v6/test6_2025-02_to_2025-05.jsonl"),
    )
    args = parser.parse_args()

    selected = []
    with args.input.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if "2025-02-00T00:00:00" <= record["contest_date"] < "2025-06-00T00:00:00":
                selected.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for record in selected:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    months = Counter(record["contest_date"][:7] for record in selected)
    print(f"input_rows={sum(1 for _ in args.input.open())}")
    print(f"selected_rows={len(selected)}")
    print(f"selected_months={dict(sorted(months.items()))}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
