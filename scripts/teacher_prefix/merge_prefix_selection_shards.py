#!/usr/bin/env python3
"""Merge sharded prefix-selection parquet outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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

    frames = []
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p))

    df = pd.concat(frames, ignore_index=True)
    if "__prefix_select_original_index" in df.columns:
        df = df.sort_values("__prefix_select_original_index").reset_index(drop=True)
        df = df.drop(columns=["__prefix_select_original_index"])

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Wrote {output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
