#!/usr/bin/env python3
"""Keep Eurus/TACO teacher trajectories that pass every provided test case."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def passes_all_tests(item: tuple[str, str]) -> bool:
    response, ground_truth = item
    # Import inside the worker so the verifier's multiprocessing/alarm state is
    # isolated from the parent parquet writer.
    from verl.utils.reward_score.prime_code import compute_score

    try:
        result = compute_score(response, ground_truth, continuous=False)
        return result[0] is True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    source = pq.ParquetFile(args.input)
    required = {"data_source", "reward_model", "teacher_response_text"}
    missing = required.difference(source.schema_arrow.names)
    if missing:
        raise RuntimeError(f"Input lacks required columns: {sorted(missing)}")
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".writing")
    if temporary.exists():
        temporary.unlink()

    writer = None
    total = correct = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch_id, batch in enumerate(source.iter_batches(batch_size=args.batch_size)):
            table = pa.Table.from_batches([batch])
            rows = table.select(["data_source", "reward_model", "teacher_response_text"]).to_pylist()
            work = []
            for row in rows:
                # Eurus-Code has several code sources (e.g. TACO,
                # CodeContests, Codeforces). Their ground truth shares the
                # same in/out contract consumed by prime_code.
                work.append((str(row["teacher_response_text"]), str(row["reward_model"]["ground_truth"])))
            keep = list(pool.map(passes_all_tests, work, chunksize=1))
            indices = [i for i, passed in enumerate(keep) if passed]
            total += table.num_rows
            correct += len(indices)
            if indices:
                selected = table.take(pa.array(indices, type=pa.int64()))
                if writer is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(temporary, selected.schema, compression="zstd")
                writer.write_table(selected)
            print(f"batch={batch_id} checked={total} correct={correct}", flush=True)

    if writer is None:
        raise RuntimeError("No correct trajectories found; no output written")
    writer.close()
    check = pq.ParquetFile(temporary)
    if check.metadata.num_rows != correct:
        raise RuntimeError(f"Output row mismatch: expected={correct}, actual={check.metadata.num_rows}")
    temporary.replace(output)
    print(f"input rows: {total}")
    print(f"correct rows: {correct}")
    print(f"correct ratio: {correct / total:.4%}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
