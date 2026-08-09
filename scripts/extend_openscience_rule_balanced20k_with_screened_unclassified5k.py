#!/usr/bin/env python3
"""Add screened unclassified examples to the science 20k candidate.

The screen is deliberately exclusion-only.  It removes questions with explicit
signals of non-target disciplines; it does not assign a science label to a
remaining question.
"""

from __future__ import annotations

import json
import re
from collections import Counter
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
OUT = (
    DATA_DIR
    / "openscience_reasoning2_science_rule_balanced20k_plus_screened_unclassified5k_"
    "mc18p75k_numeric6p25k_seed42.parquet"
)
REPORT = OUT.with_suffix(".json")

SEED = 42
QUOTAS = {"mcq": 3750, "numeric": 1250}

# Each expression is intended to identify an unambiguous non-target discipline.
# This is a conservative removal list, not a scientific-domain classifier.
EXCLUSION_PATTERNS = {
    "literature_language": re.compile(
        r"\b(?:dostoevsky|bakhtin|poetic|poetry|novel(?:s)?|formalist|"
        r"syuzhet|ostranenie|defamiliarization|literary|linguistic(?:s)?|"
        r"phonological|morphological|suffix(?:es)?|speech community|"
        r"ethnography of communication)\b",
        re.IGNORECASE,
    ),
    "history_archaeology": re.compile(
        r"\b(?:carolingian|merovingian|visigothic|medieval latin|"
        r"manuscript punctuation|pre-pottery neolithic|ppnb|ppnc|"
        r"bretton woods|fort knox)\b",
        re.IGNORECASE,
    ),
    "philosophy": re.compile(r"\b(?:goodman(?:'s)? grue|grue paradox|bleen)\b", re.IGNORECASE),
    "education_social_science": re.compile(
        r"\b(?:social learning theory|learning styles?|vark|curriculum designer|"
        r"instructional implication|educational design)\b",
        re.IGNORECASE,
    ),
    "economics_finance_business": re.compile(
        r"\b(?:monopolist|inverse demand curve|operating income|\bebit\b|"
        r"earnings per share|\beps\b|balance of payments|reserve currency|"
        r"exchange rates?|trade deficits?|savings account|compound(?:ed)? annually|"
        r"supply chain|tariff(?:s)?)\b",
        re.IGNORECASE,
    ),
    "law_policy": re.compile(
        r"\b(?:plaintiff|defendant|supreme court|statutory|tort law|"
        r"criminal code|constitutional amendment)\b",
        re.IGNORECASE,
    ),
}


def exclusion_reason(question: str) -> str | None:
    for category, pattern in EXCLUSION_PATTERNS.items():
        if pattern.search(question):
            return category
    return None


def deterministic_sample(table: pa.Table, count: int, seed: int) -> pa.Table:
    indices = table["source_index"].to_pylist()
    rank = sorted(
        range(len(indices)),
        key=lambda i: ((indices[i] * 1103515245 + seed) & 0x7FFFFFFF, indices[i]),
    )
    return table.take(pa.array(rank[:count], type=pa.int64()))


def main() -> None:
    tagged = pq.read_table(TAGGED)
    unclassified = tagged.filter(pc.equal(tagged["rule_subject"], "unclassified"))

    additions = []
    removed = Counter()
    removed_by_answer_type = {answer_type: Counter() for answer_type in QUOTAS}
    available = {}
    for offset, (answer_type, quota) in enumerate(QUOTAS.items()):
        candidates = unclassified.filter(pc.equal(unclassified["answer_type"], answer_type))
        keep_indices = []
        for index, question in enumerate(candidates["question"].to_pylist()):
            reason = exclusion_reason(question)
            if reason is None:
                keep_indices.append(index)
            else:
                removed[reason] += 1
                removed_by_answer_type[answer_type][reason] += 1
        candidates = candidates.take(pa.array(keep_indices, type=pa.int64()))
        available[answer_type] = len(candidates)
        if len(candidates) < quota:
            raise ValueError(f"Only {len(candidates)} screened {answer_type} candidates; need {quota}.")
        additions.append(deterministic_sample(candidates, quota, SEED + offset))

    addition = pa.concat_tables(additions)
    base = pq.read_table(BASE)
    combined = pa.concat_tables([base, addition])
    unique_indices = len(set(combined["source_index"].to_pylist()))
    if unique_indices != len(combined):
        raise ValueError("Duplicate source_index values in combined sample.")

    pq.write_table(combined, OUT, compression="zstd")
    answer_counts = {
        answer_type: int(pc.sum(pc.cast(pc.equal(combined["answer_type"], answer_type), pa.int64())).as_py())
        for answer_type in QUOTAS
    }
    metadata = {
        "base_sample": str(BASE),
        "addition_source": str(TAGGED),
        "method": "MCQ and numeric samples drawn after conservative explicit non-target-discipline exclusion",
        "note": "Remaining unclassified rows are not semantic science labels.",
        "seed": SEED,
        "base_rows": len(base),
        "unclassified_addition_rows": len(addition),
        "unclassified_addition_quotas": QUOTAS,
        "screened_candidate_counts": available,
        "removed_by_explicit_exclusion": dict(sorted(removed.items())),
        "removed_by_answer_type": {
            answer_type: dict(sorted(counts.items()))
            for answer_type, counts in removed_by_answer_type.items()
        },
        "rows": len(combined),
        "counts_by_answer_type": answer_counts,
        "unique_source_indices": unique_indices,
    }
    REPORT.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
