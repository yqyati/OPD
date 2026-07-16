#!/usr/bin/env python3
"""Validate exact-token alignment of cached teacher prefix top-k targets."""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--sample-rows", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pq.ParquetFile(args.input)
    columns = [
        "teacher_prefix_token_ids",
        "teacher_prefix_top_k_ids",
        "teacher_prefix_top_k_log_probs",
        "teacher_prefix_top_k_uses_generated_token_ids",
    ]
    parquet = pq.ParquetFile(args.output)
    if source.metadata.num_rows != parquet.metadata.num_rows:
        raise RuntimeError(
            f"row count mismatch: input={source.metadata.num_rows} output={parquet.metadata.num_rows}"
        )
    missing = set(columns).difference(parquet.schema_arrow.names)
    if missing:
        raise RuntimeError(f"missing top-k columns: {sorted(missing)}")

    source_batch = next(
        source.iter_batches(batch_size=args.sample_rows, columns=["teacher_prefix_token_ids"])
    )
    source_ids = source_batch.column(0).to_pylist()
    output_batch = next(parquet.iter_batches(batch_size=args.sample_rows, columns=columns))
    row_idx = 0
    for row, src_ids in zip(output_batch.to_pylist(), source_ids, strict=True):
        prefix_ids = [int(token_id) for token_id in row["teacher_prefix_token_ids"]]
        if prefix_ids != [int(token_id) for token_id in src_ids]:
            raise RuntimeError(f"row {row_idx} prefix token IDs changed during top-k caching")
        topk_ids = row["teacher_prefix_top_k_ids"]
        topk_logp = row["teacher_prefix_top_k_log_probs"]
        if len(topk_ids) != len(prefix_ids) or len(topk_logp) != len(prefix_ids):
            raise RuntimeError(
                f"row {row_idx} top-k length mismatch: prefix={len(prefix_ids)}, "
                f"ids={len(topk_ids)}, logp={len(topk_logp)}"
            )
        if any(len(values) != args.top_k for values in topk_ids):
            raise RuntimeError(f"row {row_idx} teacher top-k ID width is not {args.top_k}")
        if any(len(values) != args.top_k for values in topk_logp):
            raise RuntimeError(f"row {row_idx} teacher top-k log-prob width is not {args.top_k}")
        if not row["teacher_prefix_top_k_uses_generated_token_ids"]:
            raise RuntimeError(f"row {row_idx} top-k cache was not built from generated token IDs")
        row_idx += 1

    print(
        f"validated teacher top-k metadata: rows={parquet.metadata.num_rows}; "
        f"exact-token samples={row_idx}; k={args.top_k}"
    )


if __name__ == "__main__":
    main()
