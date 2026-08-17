#!/usr/bin/env python3
"""Stream-filter teacher-response parquet rows by generated token length."""

import argparse
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input teacher-response parquet")
    parser.add_argument("--output", required=True, help="Filtered output parquet")
    parser.add_argument("--max-response-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    reader = pq.ParquetFile(input_path)
    if "teacher_response_token_len" not in reader.schema_arrow.names:
        raise RuntimeError("Input is missing teacher_response_token_len")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_rows = 0
    dropped_rows = 0
    writer = None
    try:
        for batch in reader.iter_batches(batch_size=args.batch_size):
            keep = pc.less_equal(batch.column("teacher_response_token_len"), args.max_response_tokens)
            filtered = batch.filter(keep)
            kept_rows += filtered.num_rows
            dropped_rows += batch.num_rows - filtered.num_rows
            if filtered.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(output_path, filtered.schema, compression="snappy")
                writer.write_batch(filtered)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("All rows were filtered; no output parquet was written")
    print(
        f"input={input_path} kept={kept_rows} dropped={dropped_rows} "
        f"max_response_tokens={args.max_response_tokens} output={output_path}"
    )


if __name__ == "__main__":
    main()
