#!/usr/bin/env python3
"""Build completion-aware fixed-prefix OPD data from full teacher responses.

For responses that continue through ``prefix_length``, the saved prefix is the
first exact generated token IDs and its finish reason is recorded as ``length``:
the student rolls out after that fixed prefix.  A teacher response that stopped
before the requested boundary is retained as a complete prefix.  The RL dataset
then trains it with prefix SFT only and canonicalizes its terminal EOS when the
configured EOS bridge is enabled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Full teacher-response parquet.")
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Original pre-rollout training parquet. Required for response files "
            "whose nested prompt schema cannot be read as a complete Arrow table."
        ),
    )
    parser.add_argument("--output", required=True, help="Output teacher-prefix parquet.")
    parser.add_argument("--prefix-length", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output.")
    return parser.parse_args()


def as_token_ids(value: object, row_index: int) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Row {row_index} has no teacher_response_token_ids.")
    return [int(token_id) for token_id in value]


def main() -> None:
    args = parse_args()
    if args.prefix_length <= 0:
        raise ValueError("--prefix-length must be positive")

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite it.")

    # Eurus-Code's merged response parquet contains a nested prompt column.
    # This PyArrow build cannot read that full schema, but it *can* read the
    # flat teacher metadata plus list<int> generated IDs.  The untouched
    # prompt/metadata are therefore loaded from the original source parquet
    # below when --source is supplied.
    response_file = pq.ParquetFile(args.input)
    required = {
        "teacher_response_token_ids",
        "teacher_response_finish_reason",
        "teacher_response_model",
        "teacher_response_max_tokens",
        "teacher_response_temperature",
        "teacher_response_top_p",
        "teacher_response_enable_thinking",
    }
    missing = required.difference(response_file.schema_arrow.names)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    if args.source is None:
        raise ValueError("--source is required for streaming prefix-data construction.")
    source_path = Path(args.source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source parquet does not exist: {source_path}")
    source_file = pq.ParquetFile(source_path)
    if source_file.metadata.num_rows != response_file.metadata.num_rows:
        raise RuntimeError(
            "Source/response row mismatch: "
            f"source={source_file.metadata.num_rows}, response={response_file.metadata.num_rows}"
        )

    # Keep the nested ChatML prompt in Arrow form.  Reading and writing small,
    # synchronized batches avoids materializing 25k nested prompts and token
    # trajectories in Python at once.
    batch_size = 128
    temp_output = Path(f"{output_path}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_output.exists():
        temp_output.unlink()
    writer = None
    output_count = 0
    complete_count = 0
    rollout_count = 0
    for source_batch, response_batch in zip(
        source_file.iter_batches(batch_size=batch_size),
        response_file.iter_batches(batch_size=batch_size, columns=sorted(required)),
        strict=True,
    ):
        if source_batch.num_rows != response_batch.num_rows:
            raise RuntimeError("Source/response batch row mismatch.")
        response_rows = pa.Table.from_batches([response_batch]).to_pylist()
        prefix_ids: list[list[int]] = []
        prefix_lens: list[int] = []
        prefix_finish_reasons: list[str] = []
        for local_index, response_row in enumerate(response_rows):
            row_index = output_count + local_index
            response_ids = as_token_ids(response_row["teacher_response_token_ids"], row_index)
            response_finish_reason = str(response_row["teacher_response_finish_reason"])
            is_complete_before_boundary = len(response_ids) <= args.prefix_length and response_finish_reason == "stop"
            if is_complete_before_boundary:
                selected_ids = response_ids
                selected_finish_reason = "stop"
                complete_count += 1
            else:
                selected_ids = response_ids[: args.prefix_length]
                if len(selected_ids) != args.prefix_length:
                    raise ValueError(
                        f"Row {row_index} ended with {response_finish_reason!r} after {len(response_ids)} tokens, "
                        f"before fixed prefix boundary {args.prefix_length}."
                    )
                selected_finish_reason = "length"
                rollout_count += 1
            prefix_ids.append(selected_ids)
            prefix_lens.append(len(selected_ids))
            prefix_finish_reasons.append(selected_finish_reason)

        source_table = pa.Table.from_batches([source_batch])
        count = source_batch.num_rows
        output_table = source_table
        extra_columns = {
            "__opd_original_index": pa.array(list(range(output_count, output_count + count)), type=pa.int64()),
            "teacher_prefix_text": pa.array([""] * count, type=pa.large_string()),
            "teacher_prefix_token_ids": pa.array(prefix_ids, type=pa.list_(pa.int64())),
            "teacher_prefix_token_len": pa.array(prefix_lens, type=pa.int64()),
            "teacher_prefix_finish_reason": pa.array(prefix_finish_reasons, type=pa.large_string()),
            "teacher_prefix_model": pa.array(
                [str(row["teacher_response_model"]) for row in response_rows], type=pa.large_string()
            ),
            "teacher_prefix_max_tokens": pa.array([args.prefix_length] * count, type=pa.int64()),
            "teacher_prefix_temperature": pa.array(
                [float(row["teacher_response_temperature"]) for row in response_rows], type=pa.float64()
            ),
            "teacher_prefix_top_p": pa.array(
                [float(row["teacher_response_top_p"]) for row in response_rows], type=pa.float64()
            ),
            "teacher_prefix_enable_thinking": pa.array(
                [bool(row["teacher_response_enable_thinking"]) for row in response_rows], type=pa.bool_()
            ),
        }
        for name, values in extra_columns.items():
            if len(values) != count:
                raise RuntimeError(
                    f"Generated prefix column {name!r} has {len(values)} rows; expected {count}."
                )
            output_table = output_table.append_column(name, values)
        if writer is None:
            writer = pq.ParquetWriter(temp_output, output_table.schema, compression="snappy")
        writer.write_table(output_table)
        output_count += count

    if writer is None:
        raise RuntimeError("No rows available for prefix-data construction.")
    writer.close()
    temp_output.replace(output_path)
    if output_count != response_file.metadata.num_rows:
        raise RuntimeError("Output row count changed unexpectedly.")
    print(
        f"Wrote {output_path}: rows={output_count}, fixed_prefix_rollout_rows={rollout_count}, "
        f"completed_prefix_sft_rows={complete_count}, prefix_length={args.prefix_length}"
    )


if __name__ == "__main__":
    main()
