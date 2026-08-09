#!/usr/bin/env python3
"""Build a deterministic VERL prompt dataset from EpiCoder-func-380k.

This prepares prompts only.  The source `output` field is deliberately not
used as a training target: later SFT/OPD supervision must come from the same
configured teacher model used for OPD (teacher consistency).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/epicoder-func-380k/func_380k.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/epicoder-func-380k/epicoder_func_30k_seed42_verl.parquet"),
    )
    parser.add_argument("--sample-size", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    with args.input.open("rb") as source:
        total_rows = sum(1 for _ in source)
    if args.sample_size > total_rows:
        raise ValueError(f"sample size {args.sample_size} exceeds {total_rows} input rows")

    selected = set(random.Random(args.seed).sample(range(total_rows), args.sample_size))
    rows: list[dict[str, object]] = []
    with args.input.open(encoding="utf-8") as source:
        for original_index, line in enumerate(source):
            if original_index not in selected:
                continue
            record = json.loads(line)
            instruction = record.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"row {original_index} has no non-empty instruction")
            rows.append(
                {
                    "data_source": "epicoder_func_380k",
                    "prompt": [{"role": "user", "content": instruction}],
                    "ability": "code",
                    # EpiCoder has reference implementations but no executable
                    # verifier payload compatible with the Eurus code manager.
                    # The planned OPD run uses teacher token rewards only.
                    "reward_model": {"style": "none", "ground_truth": ""},
                    "extra_info": {"index": original_index},
                }
            )

    if len(rows) != args.sample_size:
        raise RuntimeError(f"selected {len(rows)} rows, expected {args.sample_size}")

    schema = pa.schema(
        [
            ("data_source", pa.string()),
            (
                "prompt",
                pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
            ),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            ("extra_info", pa.struct([("index", pa.int64())])),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    metadata = {
        b"source_dataset": b"microsoft/EpiCoder-func-380k",
        b"source_file": b"func_380k.jsonl",
        b"source_rows": str(total_rows).encode(),
        b"sample_size": str(args.sample_size).encode(),
        b"sample_seed": str(args.seed).encode(),
        b"selection": b"random.Random(seed).sample(range(source_rows), sample_size)",
        b"source_output_used_as_target": b"false",
    }
    table = table.replace_schema_metadata(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd")
    print(f"wrote {table.num_rows} rows to {args.output}")


if __name__ == "__main__":
    main()
