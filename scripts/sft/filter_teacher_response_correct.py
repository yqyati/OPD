#!/usr/bin/env python3
"""Filter teacher-response parquet to rows where teacher response is correct."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "verl"))

from verl.utils.reward_score.ttrl_math import compute_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--response-column", default="teacher_response_text")
    parser.add_argument("--ground-truth-column", default="reward_model")
    parser.add_argument(
        "--require-stop",
        action="store_true",
        help="Only retain naturally stopped generations.",
    )
    parser.add_argument(
        "--require-complete-thinking",
        action="store_true",
        help="Only retain responses containing a closing </think> tag.",
    )
    return parser.parse_args()


def extract_ground_truth(row: pd.Series, column: str):
    value = row[column]
    if isinstance(value, dict):
        return value.get("ground_truth")
    if hasattr(value, "get"):
        return value.get("ground_truth")
    return value


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)

    keep = []
    scores = []
    format_scores = []
    for _, row in df.iterrows():
        response = row.get(args.response_column, "")
        if args.require_stop and row.get("teacher_response_finish_reason") != "stop":
            keep.append(False)
            scores.append(0.0)
            format_scores.append(0.0)
            continue
        if args.require_complete_thinking and "</think>" not in str(response):
            keep.append(False)
            scores.append(0.0)
            format_scores.append(0.0)
            continue
        gt = extract_ground_truth(row, args.ground_truth_column)
        result = compute_score(str(response), str(gt), fast=False)
        acc = bool(result.get("acc", False))
        keep.append(acc)
        scores.append(float(result.get("score", 0.0)))
        format_scores.append(float(result.get("format_score", 0.0)))

    out_df = df.loc[keep].copy()
    out_df["teacher_response_filter_score"] = [s for s, k in zip(scores, keep) if k]
    out_df["teacher_response_filter_format_score"] = [s for s, k in zip(format_scores, keep) if k]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)

    total = len(df)
    correct = len(out_df)
    print(f"input rows: {total}")
    print(f"correct rows: {correct}")
    print(f"correct ratio: {correct / total:.4%}" if total else "correct ratio: n/a")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
