#!/usr/bin/env python3
"""Create the science-aligned SuperGPQA split from native metadata labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "test_data" / "SuperGPQA"
SOURCE = DATA_DIR / "SuperGPQA-all.jsonl"
OUTPUT = DATA_DIR / "SuperGPQA-science-engineering-medicine.jsonl"
REPORT = DATA_DIR / "SuperGPQA-science-engineering-medicine.json"
KEEP_DISCIPLINES = {"Science", "Engineering", "Medicine"}


def main() -> None:
    retained = []
    all_disciplines = Counter()
    retained_disciplines = Counter()
    retained_fields = Counter()
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            discipline = row["discipline"]
            all_disciplines[discipline] += 1
            if discipline in KEEP_DISCIPLINES:
                retained.append(row)
                retained_disciplines[discipline] += 1
                retained_fields[row["field"]] += 1

    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": str(SOURCE),
        "method": "native discipline metadata only; no question-text matching",
        "kept_disciplines": sorted(KEEP_DISCIPLINES),
        "source_rows": sum(all_disciplines.values()),
        "rows": len(retained),
        "excluded_rows": sum(all_disciplines.values()) - len(retained),
        "all_discipline_counts": dict(sorted(all_disciplines.items())),
        "retained_discipline_counts": dict(sorted(retained_disciplines.items())),
        "retained_field_counts": dict(sorted(retained_fields.items())),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
