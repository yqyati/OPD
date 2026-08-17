#!/usr/bin/env python3
"""Validate and merge four independent LCB generation shards by question ID."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("shards", nargs="+", help="LCB generation JSON files")
    args = parser.parse_args()

    merged: dict[str, dict] = {}
    for shard_name in args.shards:
        with Path(shard_name).open() as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise RuntimeError(f"Expected a JSON list in {shard_name}")
        for row in rows:
            question_id = str(row["question_id"])
            if question_id in merged:
                raise RuntimeError(f"Duplicate question_id {question_id} in {shard_name}")
            if not row.get("output_list"):
                raise RuntimeError(f"Missing generations for question_id {question_id}")
            merged[question_id] = row

    if not merged:
        raise RuntimeError("No LCB generations found in shard files")

    output_rows = [merged[key] for key in sorted(merged)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(output_rows, handle, indent=2)
    print(f"Merged {len(output_rows)} LCB problems into {output}")


if __name__ == "__main__":
    main()
