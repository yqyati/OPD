#!/usr/bin/env python3
"""Merge prompt OPD scoring shards and write filtered subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from score_prompt_opd_data import add_scores, write_subsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--top-fracs", default="0.5,0.3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    parts = []
    for shard_idx in range(args.num_shards):
        path = input_dir / f"opd_prompt_scores_shard{shard_idx}of{args.num_shards}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path))

    scored = pd.concat(parts, ignore_index=True)
    if "__opd_original_index" in scored.columns:
        scored = scored.sort_values("__opd_original_index").reset_index(drop=True)
    scored = add_scores(scored, args.topk)

    scored_path = input_dir / "opd_prompt_scores.parquet"
    scored.to_parquet(scored_path, index=False)
    print(f"Wrote {scored_path} ({len(scored)} rows)")

    top_fracs = [float(x) for x in args.top_fracs.split(",") if x.strip()]
    write_subsets(scored, input_dir, top_fracs, args.topk)


if __name__ == "__main__":
    main()
