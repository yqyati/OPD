#!/usr/bin/env python3
"""Strictly merge sharded teacher-handoff annotation parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = [input_dir / f"handoff_annotation_shard_{shard_id:02d}_of_{args.num_shards:02d}.parquet" for shard_id in range(args.num_shards)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing annotation shards: {missing}")

    frames = [pd.read_parquet(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True)
    if "teacher_handoff_annotation_row_id" not in merged.columns:
        raise RuntimeError("annotation shards are missing teacher_handoff_annotation_row_id")
    if merged["teacher_handoff_annotation_row_id"].duplicated().any():
        raise RuntimeError("duplicate source row IDs across annotation shards")
    merged.sort_values("teacher_handoff_annotation_row_id", inplace=True)
    expected = list(range(len(merged)))
    observed = merged["teacher_handoff_annotation_row_id"].astype(int).tolist()
    if observed != expected:
        raise RuntimeError("annotation shard row IDs are not a complete contiguous source dataset")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    ok = merged[merged["teacher_handoff_annotation_status"] == "ok"]
    stats = {
        "rows": len(merged),
        "ok": len(ok),
        "parse_rate": len(ok) / len(merged) if len(merged) else 0.0,
        "budget_quantiles": ok["teacher_handoff_token_budget"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
        "budget_counts": ok["teacher_handoff_token_budget"].value_counts().sort_index().to_dict(),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
