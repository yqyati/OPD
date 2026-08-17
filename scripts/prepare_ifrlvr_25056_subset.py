#!/usr/bin/env python3
"""Create a deterministic 25,056-example IF-RLVR training subset.

The source dataset is AllenAI's ``allenai/IF_multi_constraints_upto5``.
It records all rows as ``constraint_type=multi``; the useful difficulty
stratum is therefore derived from the number of tab-separated constraints.

The default allocation deliberately balances 1--5 constraint instructions:
    {1: 5011, 2: 5011, 3: 5011, 4: 5011, 5: 5012}
This is preferred to proportional sampling because five-constraint examples
are the scarce, hardest cases and would otherwise account for only 1,866 of
25,056 examples. 25,056 also equals 261 training steps x batch size 96.

The script does *not* download data or launch training. It writes a selected
Parquet file, selected source keys, and an audit manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


TARGET_ROWS = 25_056
DEFAULT_SEED = 42
BALANCED_ALLOCATION = {1: 5_011, 2: 5_011, 3: 5_011, 4: 5_011, 5: 5_012}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def constraint_count(row: dict[str, Any]) -> int:
    constraint = row.get("constraint")
    if not isinstance(constraint, str) or not constraint.strip():
        raise ValueError(f"Empty/malformed constraint for key={row.get('key')!r}")
    return constraint.count("\t") + 1


def choose_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[int, int]]:
    # The five duplicate source keys are excluded deterministically. Key is the
    # dataset identifier used for downstream auditing, so selected rows must be
    # uniquely addressable.
    unique_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in rows:
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"Missing/malformed key: {key!r}")
        if key not in seen_keys:
            unique_rows.append(row)
            seen_keys.add(key)

    buckets: dict[int, list[dict[str, Any]]] = {n: [] for n in BALANCED_ALLOCATION}
    for row in unique_rows:
        n_constraints = constraint_count(row)
        if n_constraints not in buckets:
            raise ValueError(f"Expected 1--5 constraints, got {n_constraints} for {row['key']}")
        buckets[n_constraints].append(row)

    chosen: list[dict[str, Any]] = []
    available = {n: len(bucket) for n, bucket in buckets.items()}
    for n_constraints, requested in BALANCED_ALLOCATION.items():
        bucket = list(buckets[n_constraints])
        random.Random(seed + 10_000 * n_constraints).shuffle(bucket)
        if len(bucket) < requested:
            raise ValueError(
                f"Stratum {n_constraints} has {len(bucket)} rows but needs {requested}."
            )
        chosen.extend(bucket[:requested])

    # Shuffle the assembled data once more. Training sees a mixed-difficulty
    # stream while allocation remains exactly balanced.
    random.Random(seed).shuffle(chosen)
    if len(chosen) != TARGET_ROWS or len({row['key'] for row in chosen}) != TARGET_ROWS:
        raise AssertionError("Selection is not exactly TARGET_ROWS unique keys.")
    return chosen, available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Input train Parquet from IF-RLVR.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for subset and audit files.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_table = pq.read_table(source)
    required = {"key", "constraint", "messages", "ground_truth", "dataset", "constraint_type"}
    missing = required - set(source_table.column_names)
    if missing:
        raise ValueError(f"Unexpected source schema; missing {sorted(missing)}")
    source_rows = source_table.to_pylist()
    selected, available = choose_rows(source_rows, args.seed)

    subset_path = output_dir / "ifrlvr_train_25056_balanced_constraints_seed42.parquet"
    key_path = output_dir / "ifrlvr_train_25056_balanced_constraints_seed42_keys.json"
    manifest_path = output_dir / "ifrlvr_train_25056_balanced_constraints_seed42_manifest.json"
    pq.write_table(pa.Table.from_pylist(selected, schema=source_table.schema), subset_path, compression="zstd")
    keys = [row["key"] for row in selected]
    key_path.write_text(json.dumps(keys, ensure_ascii=False, indent=2) + "\n")

    selected_counts = Counter(constraint_count(row) for row in selected)
    manifest = {
        "dataset": "allenai/IF_multi_constraints_upto5",
        "selection_policy": "balanced by derived number of tab-separated constraints",
        "target_rows": TARGET_ROWS,
        "seed": args.seed,
        "source": str(source),
        "source_sha256": sha256(source),
        "source_rows": len(source_rows),
        "source_unique_keys": len({row["key"] for row in source_rows}),
        "excluded_duplicate_key_rows": len(source_rows) - len({row["key"] for row in source_rows}),
        "available_unique_rows_by_constraint_count": available,
        "requested_rows_by_constraint_count": BALANCED_ALLOCATION,
        "selected_rows_by_constraint_count": dict(sorted(selected_counts.items())),
        "outputs": {"subset": str(subset_path), "keys": str(key_path)},
        "notes": [
            "Constraint count is constraint.count('\\t') + 1.",
            "Do not train on IFBench_test or IFBench_multi-turn; they are evaluation sets.",
            "The released training schema has no direct IFEval-vs-IFBench-Train source column.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(selected)} examples: {subset_path}")
    print(f"Constraint-count allocation: {dict(sorted(selected_counts.items()))}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
