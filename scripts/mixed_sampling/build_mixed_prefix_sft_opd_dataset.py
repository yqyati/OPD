#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a batch-level mixed teacher-SFT + Plain-OPD parquet."
    )
    parser.add_argument("--input", required=True, help="Teacher target parquet.")
    parser.add_argument("--output", required=True, help="Output mixed parquet.")
    parser.add_argument(
        "--target-column",
        default="teacher_prefix_text",
        help="Column to use as teacher SFT target. Use teacher_response_text for full-response SFT.",
    )
    parser.add_argument(
        "--teacher-every",
        type=int,
        default=2,
        help="Mark one row as teacher_sft every N rows. Default 2 gives a 1:1 teacher/student mix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.teacher_every <= 1:
        raise ValueError("--teacher-every must be > 1.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")

    df = pd.read_parquet(input_path)
    if args.target_column not in df.columns:
        raise ValueError(f"{input_path} does not contain {args.target_column}.")

    mixed = df.copy()
    is_teacher = mixed.index.to_series().mod(args.teacher_every).eq(0)

    mixed["teacher_prefix_text"] = ""
    mixed.loc[is_teacher, "teacher_prefix_text"] = mixed.loc[is_teacher, args.target_column].fillna("").astype(str)

    mixed["mixed_sample_type"] = "student_opd"
    mixed.loc[is_teacher, "mixed_sample_type"] = "teacher_sft"

    mixed["opd_loss_mask"] = 1.0
    mixed.loc[is_teacher, "opd_loss_mask"] = 0.0

    mixed["mixed_sft_target_column"] = args.target_column

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_parquet(output_path, index=False)

    teacher_count = int(is_teacher.sum())
    student_count = int((~is_teacher).sum())
    print(f"Wrote {output_path}")
    print(f"rows={len(mixed)} teacher_sft={teacher_count} student_opd={student_count}")


if __name__ == "__main__":
    main()
