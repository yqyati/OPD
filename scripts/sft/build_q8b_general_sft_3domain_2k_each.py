#!/usr/bin/env python3
"""Build the fixed, shared Stage-1 SFT dataset for the strict 8B MOPD line.

This intentionally samples from the three *already materialized exact-token SFT*
parquets.  It does not sample RL prompts and never regenerates teacher rollouts.
The output retains the precomputed input IDs and loss masks used by
``PrecomputedTokenSFTDataset``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DOMAINS = ("math", "code", "instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--instruct", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-domain", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_domain(path: Path, domain: str, per_domain: int, seed: int) -> pa.Table:
    table = pq.read_table(path)
    required = {
        "enable_thinking",
        "sft_token_len",
        "precomputed_input_ids",
        "precomputed_loss_mask",
        "sft_uses_generated_token_ids",
    }
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"{path}: missing required SFT fields: {sorted(missing)}")
    if table.num_rows < per_domain:
        raise ValueError(f"{path}: has {table.num_rows} rows, need {per_domain}")

    # Independent fixed streams avoid a domain's sample depending on input order.
    domain_seed = seed + {"math": 0, "code": 10_000, "instruct": 20_000}[domain]
    indices = np.random.default_rng(domain_seed).choice(table.num_rows, size=per_domain, replace=False)
    indices.sort()
    sampled = table.take(pa.array(indices, type=pa.int64()))
    sampled = sampled.append_column("mopd_domain", pa.array([domain] * per_domain))
    sampled = sampled.append_column("mopd_source_row", pa.array(indices, type=pa.int64()))
    return sampled


def main() -> None:
    args = parse_args()
    if args.per_domain <= 0:
        raise ValueError("--per-domain must be positive")
    paths = {
        "math": Path(args.math),
        "code": Path(args.code),
        "instruct": Path(args.instruct),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    sampled = [sample_domain(paths[domain], domain, args.per_domain, args.seed) for domain in DOMAINS]
    merged = pa.concat_tables(sampled, promote_options="none")

    # Shuffle the final manifest once, while preserving exact per-domain counts.
    permutation = np.random.default_rng(args.seed).permutation(merged.num_rows)
    merged = merged.take(pa.array(permutation, type=pa.int64()))
    expected_rows = args.per_domain * len(DOMAINS)
    if merged.num_rows != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, produced {merged.num_rows}")
    counts = {
        domain: int(pc.sum(pc.equal(merged["mopd_domain"], domain)).as_py())
        for domain in DOMAINS
    }
    if any(count != args.per_domain for count in counts.values()):
        raise RuntimeError(f"Domain-count invariant failed: {counts}")
    if not bool(pc.all(merged["sft_uses_generated_token_ids"]).as_py()):
        raise RuntimeError("Every selected row must use saved generated token IDs")

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(merged, tmp_output, compression="zstd")
    tmp_output.replace(output)
    print(f"wrote: {output}")
    print(f"rows: {merged.num_rows}; per-domain: {counts}; seed: {args.seed}")
    lengths = merged["sft_token_len"].to_numpy()
    print(
        "sft_token_len min/mean/p99/max: "
        f"{lengths.min()} / {lengths.mean():.2f} / {np.quantile(lengths, 0.99):.0f} / {lengths.max()}"
    )


if __name__ == "__main__":
    main()
