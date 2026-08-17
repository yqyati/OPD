#!/usr/bin/env python3
"""Merge sharded prefix-selection parquet outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"{output} already exists. Use --force to overwrite.")
        return

    tables = []
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        tables.append(parquet.read_table(p))

    table = pa.concat_tables(tables, promote_options="default")
    sort_column = None
    if "__prefix_select_original_index" in table.column_names:
        sort_column = "__prefix_select_original_index"
    elif "__opd_original_index" in table.column_names:
        sort_column = "__opd_original_index"
    if sort_column is not None:
        order = sorted(range(table.num_rows), key=lambda i: table[sort_column][i].as_py())
        table = table.take(pa.array(order, type=pa.int64())).drop([sort_column])

    output.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, output, compression="zstd")
    print(f"Wrote {output} ({table.num_rows} rows)")


if __name__ == "__main__":
    main()
