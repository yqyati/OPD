#!/usr/bin/env python3
"""Build Soft Teachability Sampling subsets from scored OPD prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input scored parquet with opd_prompt_score.")
    parser.add_argument("--output", required=True, help="Output subset parquet.")
    parser.add_argument("--score-column", default="opd_prompt_score")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument(
        "--sort-output",
        action="store_true",
        help="Sort output by score descending. Default keeps original dataset order.",
    )
    return parser.parse_args()


def allocate_counts(bin_sizes: list[int], target_size: int, alpha: float) -> list[int]:
    if target_size > sum(bin_sizes):
        raise ValueError("target_size cannot exceed dataset size")
    if len(bin_sizes) == 1:
        return [target_size]

    # Normalize bin rank to [0, 1]. This keeps alpha=0.5/1.0 as a soft bias.
    ranks = np.linspace(0.0, 1.0, len(bin_sizes), dtype=np.float64)
    weights = np.exp(alpha * ranks)
    raw = target_size * weights / weights.sum()

    counts = np.floor(raw).astype(int)
    caps = np.asarray(bin_sizes, dtype=int)
    counts = np.minimum(counts, caps)

    remaining = target_size - int(counts.sum())
    remainders = raw - np.floor(raw)
    while remaining > 0:
        available = np.where(counts < caps)[0]
        if available.size == 0:
            raise RuntimeError("No available bins left while allocating counts.")
        best = available[np.argmax(remainders[available])]
        counts[best] += 1
        remainders[best] = 0.0
        remaining -= 1

    return counts.tolist()


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1].")
    if args.num_bins < 1:
        raise ValueError("--num-bins must be >= 1.")

    df = pd.read_parquet(args.input)
    if args.score_column not in df.columns:
        raise ValueError(f"Missing score column: {args.score_column}")

    target_size = max(1, int(len(df) * args.fraction))
    sorted_df = df.sort_values(args.score_column, ascending=True).reset_index(drop=True)
    bins = np.array_split(sorted_df.index.to_numpy(), args.num_bins)
    bin_sizes = [len(indices) for indices in bins]
    counts = allocate_counts(bin_sizes, target_size, args.alpha)

    rng = np.random.default_rng(args.seed)
    chosen_parts = []
    for indices, count in zip(bins, counts, strict=True):
        if count == 0:
            continue
        chosen = rng.choice(indices, size=count, replace=False)
        chosen_parts.append(sorted_df.iloc[np.sort(chosen)])

    subset = pd.concat(chosen_parts, ignore_index=True)
    if args.sort_output:
        subset = subset.sort_values(args.score_column, ascending=False).reset_index(drop=True)
    elif "__opd_original_index" in subset.columns:
        subset = subset.sort_values("__opd_original_index").reset_index(drop=True)
    else:
        subset = subset.sort_index().reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output, index=False)

    print(f"Wrote {output} ({len(subset)} rows)")
    print("Bin sizes:", bin_sizes)
    print("Sample counts:", counts)
    print(
        subset[args.score_column]
        .describe(percentiles=[0.1, 0.2, 0.5, 0.8, 0.9])
        .to_string()
    )


if __name__ == "__main__":
    main()
