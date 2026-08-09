#!/usr/bin/env python3
"""Extract OSR-2 rows with exact MCQ-letter or pure-numeric answers.

This intentionally performs no subject, domain, quality, or semantic filtering.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests


SOURCE = (
    "https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2/resolve/"
    "main/train/OpenScienceReasoning-2.parquet"
)
MCQ = re.compile(r"^\s*(?:\\boxed\{)?([A-J])\}?\s*$", re.IGNORECASE)
NUMERIC = re.compile(r"^\s*[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:\s*(?:%|[A-Za-zµμ°]+))?\s*$")


class HttpRangeReader(io.RawIOBase):
    def __init__(self, source: str) -> None:
        self.session = requests.Session()
        response = self.session.head(source, allow_redirects=True, timeout=120)
        response.raise_for_status()
        self.source = response.url
        self.size = int(response.headers["content-length"])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def readinto(self, buffer: Any) -> int:
        if self.position >= self.size:
            return 0
        end = min(self.size - 1, self.position + len(buffer) - 1)
        response = self.session.get(
            self.source,
            headers={"Range": f"bytes={self.position}-{end}"},
            timeout=240,
        )
        response.raise_for_status()
        data = response.content[: len(buffer)]
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def answer_type(answer: str) -> tuple[str, str] | None:
    answer = answer.strip()
    match = MCQ.match(answer)
    if match:
        return "mcq", match.group(1).upper()
    if NUMERIC.match(answer):
        return "numeric", answer
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-group", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parquet = pq.ParquetFile(HttpRangeReader(SOURCE))
    if not 0 <= args.row_group < parquet.metadata.num_row_groups:
        raise ValueError(f"row group must be in [0, {parquet.metadata.num_row_groups})")
    table = parquet.read_row_group(args.row_group, columns=["input", "expected_answer"])
    rows: list[dict[str, Any]] = []
    offset = sum(parquet.metadata.row_group(i).num_rows for i in range(args.row_group))
    for local_index, (question, answer) in enumerate(
        zip(table.column("input").to_pylist(), table.column("expected_answer").to_pylist(), strict=True)
    ):
        kind = answer_type(answer)
        if kind is None:
            continue
        category, normalized_answer = kind
        rows.append(
            {
                "source_index": offset + local_index,
                "question": question,
                "expected_answer": normalized_answer,
                "answer_type": category,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"verifiable_rg{args.row_group:02d}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd")
    print(f"row_group={args.row_group} source_rows={table.num_rows} selected_rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()
