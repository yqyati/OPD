#!/usr/bin/env python3
"""Extend the science 20k sample with a 3:1 MCQ/numeric unclassified sample."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "openscience_reasoning2"
TAGGED = (
    DATA_DIR
    / "openscience_reasoning2_verifiable_mcq45k_numeric15k_60k_seed42_"
    "exclude_humanities_econ_rule_tags.parquet"
)
BASE = DATA_DIR / "openscience_reasoning2_science_rule_balanced20k_mc15k_numeric5k_seed42.parquet"
OUT = DATA_DIR / "openscience_reasoning2_science_rule_balanced20k_plus_unclassified5k_mc18p75k_numeric6p25k_seed42.parquet"
REPORT = OUT.with_suffix(".json")

SEED = 42
QUOTAS = {"mcq": 3750, "numeric": 1250}


def sample(table: pa.Table, count: int, seed: int) -> pa.Table:
    """Use a deterministic pseudo-random ordering based on source indices."""
    indices = table["source_index"].to_pylist()
    ranked = sorted(range(len(indices)), key=lambda i: ((indices[i] * 1103515245 + seed) & 0x7FFFFFFF, indices[i]))
    return table.take(pa.array(ranked[:count], type=pa.int64()))


def main() -> None:
    tagged = pq.read_table(TAGGED)
    unclassified = tagged.filter(pc.equal(tagged["rule_subject"], "unclassified"))

    additions = []
    for offset, (answer_type, quota) in enumerate(QUOTAS.items()):
        candidates = unclassified.filter(pc.equal(unclassified["answer_type"], answer_type))
        if len(candidates) < quota:
            raise ValueError(f"Only {len(candidates)} unclassified {answer_type} rows available; need {quota}.")
        additions.append(sample(candidates, quota, SEED + offset))

    addition = pa.concat_tables(additions)
    base = pq.read_table(BASE)
    combined = pa.concat_tables([base, addition])

    unique_indices = len(set(combined["source_index"].to_pylist()))
    if unique_indices != len(combined):
        raise ValueError("Duplicate source_index values in combined sample.")

    pq.write_table(combined, OUT, compression="zstd")
    counts = {answer_type: int(pc.sum(pc.cast(pc.equal(combined["answer_type"], answer_type), pa.int64())).as_py()) for answer_type in QUOTAS}
    metadata = {
        "base_sample": str(BASE),
        "addition_source": str(TAGGED),
        "method": "deterministic random sample from rows tagged unclassified; no additional subject rules applied",
        "seed": SEED,
        "base_rows": len(base),
        "unclassified_addition_rows": len(addition),
        "unclassified_addition_quotas": QUOTAS,
        "rows": len(combined),
        "counts_by_answer_type": counts,
        "unique_source_indices": unique_indices,
    }
    REPORT.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
