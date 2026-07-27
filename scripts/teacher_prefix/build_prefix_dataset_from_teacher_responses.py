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

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Full teacher-response parquet.")
    parser.add_argument("--output", required=True, help="Output teacher-prefix parquet.")
    parser.add_argument("--prefix-length", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output.")
    return parser.parse_args()


def as_token_ids(value: object, row_index: int) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
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

    dataframe = pd.read_parquet(args.input).reset_index(drop=True)
    required = {
        "teacher_response_token_ids",
        "teacher_response_finish_reason",
        "teacher_response_model",
        "teacher_response_max_tokens",
        "teacher_response_temperature",
        "teacher_response_top_p",
        "teacher_response_enable_thinking",
    }
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    prefix_ids: list[list[int]] = []
    prefix_lens: list[int] = []
    prefix_finish_reasons: list[str] = []
    prefix_texts: list[str] = []
    complete_count = 0
    rollout_count = 0

    for row_index, row in dataframe.iterrows():
        response_ids = as_token_ids(row["teacher_response_token_ids"], row_index)
        response_finish_reason = str(row["teacher_response_finish_reason"])
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
            # This is a truncated *prefix*, even if the original full response
            # later ended by stop.  Marking it stop would incorrectly suppress
            # suffix OPD in RLHFDataset.
            selected_finish_reason = "length"
            rollout_count += 1

        prefix_ids.append(selected_ids)
        prefix_lens.append(len(selected_ids))
        prefix_finish_reasons.append(selected_finish_reason)
        # Token IDs, not this display column, are the training contract.
        prefix_texts.append("")

    output = dataframe.drop(
        columns=[column for column in dataframe.columns if column.startswith("teacher_response_")]
    ).copy()
    output["__opd_original_index"] = range(len(output))
    output["teacher_prefix_text"] = prefix_texts
    output["teacher_prefix_token_ids"] = prefix_ids
    output["teacher_prefix_token_len"] = prefix_lens
    output["teacher_prefix_finish_reason"] = prefix_finish_reasons
    output["teacher_prefix_model"] = dataframe["teacher_response_model"]
    output["teacher_prefix_max_tokens"] = args.prefix_length
    output["teacher_prefix_temperature"] = dataframe["teacher_response_temperature"]
    output["teacher_prefix_top_p"] = dataframe["teacher_response_top_p"]
    output["teacher_prefix_enable_thinking"] = dataframe["teacher_response_enable_thinking"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    if len(output) != len(dataframe):
        raise RuntimeError("Output row count changed unexpectedly.")
    if any(length <= 0 or length > args.prefix_length for length in prefix_lens):
        raise RuntimeError("Invalid output prefix lengths.")
    print(
        f"Wrote {output_path}: rows={len(output)}, fixed_prefix_rollout_rows={rollout_count}, "
        f"completed_prefix_sft_rows={complete_count}, prefix_length={args.prefix_length}"
    )


if __name__ == "__main__":
    main()
