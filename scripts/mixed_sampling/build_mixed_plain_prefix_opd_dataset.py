#!/usr/bin/env python3
"""Build a balanced sample-level plain-OPD/prefix-OPD dataset.

Prefix rows retain the exact token-ID prefix produced by the teacher.  Plain
rows have every teacher-prefix field cleared, so RLHFDataset treats them as
ordinary prompt-only OPD examples.  A seeded permutation selects exactly half
of the rows for each branch without duplicating data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-fraction", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 < args.prefix_fraction < 1.0:
        raise ValueError("--prefix-fraction must be strictly between 0 and 1")
    source = Path(args.input)
    output = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; remove it to rebuild.")

    df = pd.read_parquet(source).reset_index(drop=True)
    required = {"teacher_prefix_token_ids", "teacher_prefix_finish_reason"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing prefix contract columns: {sorted(missing)}")

    n_rows = len(df)
    n_prefix = int(round(n_rows * args.prefix_fraction))
    rng = np.random.default_rng(args.seed)
    prefix_indices = set(rng.permutation(n_rows)[:n_prefix].tolist())
    is_prefix = np.fromiter((i in prefix_indices for i in range(n_rows)), dtype=bool, count=n_rows)

    # Every row remains eligible for suffix OPD.  Completion-aware handling in
    # RLHFDataset will turn the very short natural-EOS prefix rows into SFT-only
    # rows, exactly as in the standalone prefix run.
    df["opd_loss_mask"] = 1.0
    df["mixed_sample_type"] = np.where(is_prefix, "prefix_opd", "plain_opd")
    df["mixed_prefix_seed"] = args.seed

    # Plain rows must have no teacher prefix at all.  Clear all prefix-related
    # fields, including optional top-k logits, to prevent accidental leakage.
    prefix_columns = [c for c in df.columns if c.startswith("teacher_prefix_")]
    for column in prefix_columns:
        if column == "teacher_prefix_token_ids":
            # ``.loc[..., col] = [[], ...]`` is interpreted as a 2-D ndarray
            # by recent pandas.  ``.at`` preserves each empty list as one
            # Arrow list cell.
            for row_index in df.index[~is_prefix]:
                df.at[row_index, column] = []
        elif column in {"teacher_prefix_text", "teacher_prefix_finish_reason"}:
            df.loc[~is_prefix, column] = ""
        elif column.endswith("_token_len") or column.endswith("_max_tokens"):
            df.loc[~is_prefix, column] = 0
        elif column.endswith("_enable_thinking"):
            df.loc[~is_prefix, column] = False
        elif column.endswith("_temperature") or column.endswith("_top_p"):
            df.loc[~is_prefix, column] = np.nan
        else:
            # Model names and any future prefix metadata are not used for
            # plain rows; empty strings are safest for Arrow string columns.
            df.loc[~is_prefix, column] = ""

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(
        f"Wrote {output}: rows={n_rows} prefix_opd={int(is_prefix.sum())} "
        f"plain_opd={int((~is_prefix).sum())} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
