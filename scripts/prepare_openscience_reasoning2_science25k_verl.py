#!/usr/bin/env python3
"""Convert the selected OpenScienceReasoning-2 science 25k into verl RL schema."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "openscience_reasoning2"
SOURCE = (
    DATA_DIR
    / "openscience_reasoning2_science_rule_balanced20k_plus_screened_unclassified5k_"
    "mc18p75k_numeric6p25k_seed42.parquet"
)
OUTPUT = DATA_DIR / "openscience_reasoning2_science25k_mc18p75k_numeric6p25k_seed42_verl.parquet"


def main() -> None:
    source = pq.read_table(SOURCE, columns=["source_index", "question", "expected_answer", "answer_type", "rule_subject"])
    rows = []
    for row in source.to_pylist():
        rows.append(
            {
                "data_source": "openscience_reasoning2_science",
                "prompt": [{"role": "user", "content": row["question"]}],
                "ability": "science",
                "reward_model": {"ground_truth": row["expected_answer"], "style": "rule_science"},
                "extra_info": {
                    "index": str(row["source_index"]),
                    "answer_type": row["answer_type"],
                    "rule_subject": row["rule_subject"],
                },
            }
        )

    table = pa.Table.from_pylist(rows)
    if len(table) != 25000:
        raise ValueError(f"Expected 25,000 rows, got {len(table)}")
    pq.write_table(table, OUTPUT, compression="zstd")
    print(f"Wrote {len(table)} rows to {OUTPUT}")
    print(table.schema)


if __name__ == "__main__":
    main()
