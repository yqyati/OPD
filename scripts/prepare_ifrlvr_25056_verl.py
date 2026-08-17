#!/usr/bin/env python3
"""Convert the audited IF-RLVR subset to the verl GRPO Parquet schema."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "ifrlvr"
SOURCE = DATA_DIR / "ifrlvr_train_25056_balanced_constraints_seed42.parquet"
OUTPUT = DATA_DIR / "ifrlvr_train_25056_balanced_constraints_seed42_verl.parquet"


def main() -> None:
    source = pq.read_table(SOURCE, columns=["key", "messages", "ground_truth", "constraint"])
    rows = []
    for row in source.to_pylist():
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Malformed messages for {row['key']!r}")
        rows.append(
            {
                "data_source": "ifrlvr",
                "prompt": messages,
                "ability": "instruction_following",
                "reward_model": {"ground_truth": row["ground_truth"], "style": "rule_ifrlvr"},
                "extra_info": {
                    "key": row["key"],
                    "constraint_text": row["constraint"],
                    "constraint_count": row["constraint"].count("\t") + 1,
                },
            }
        )
    if len(rows) != 25_056:
        raise ValueError(f"Expected 25,056 rows, got {len(rows)}")
    if len({row["extra_info"]["key"] for row in rows}) != len(rows):
        raise ValueError("Expected unique source keys")
    pq.write_table(pa.Table.from_pylist(rows), OUTPUT, compression="zstd")
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
