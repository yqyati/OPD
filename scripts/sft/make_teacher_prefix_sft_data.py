#!/usr/bin/env python3
"""Build pure-SFT data from teacher-response OPD parquet.

Each row becomes a single-turn conversation:
  user: original prompt
  assistant: selected teacher response column

Only assistant tokens are supervised by verl's MultiTurnSFTDataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--response-column", default="teacher_prefix_text")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def normalize_prompt(prompt) -> list[dict]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"Expected non-empty prompt list, got {type(prompt)}")
    return [dict(m) for m in prompt]


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    df = pd.read_parquet(args.input)

    rows = []
    lengths = []
    skipped_empty = 0
    skipped_long = 0
    for _, row in df.iterrows():
        teacher_response = row.get(args.response_column, "")
        if teacher_response is None or str(teacher_response) == "":
            skipped_empty += 1
            continue

        messages = normalize_prompt(row["prompt"])
        messages.append({"role": "assistant", "content": str(teacher_response)})

        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=args.enable_thinking,
        )
        length = len(token_ids)
        if length > args.max_length:
            skipped_long += 1
            continue

        out_row = row.to_dict()
        out_row["messages"] = messages
        out_row["enable_thinking"] = bool(args.enable_thinking)
        out_row["sft_token_len"] = length
        rows.append(out_row)
        lengths.append(length)

    if not rows:
        raise RuntimeError("No rows left after teacher-prefix SFT conversion.")

    out_df = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)

    lengths_sorted = sorted(lengths)
    p99 = lengths_sorted[int(0.99 * (len(lengths_sorted) - 1))]
    p999 = lengths_sorted[int(0.999 * (len(lengths_sorted) - 1))]
    print(f"input rows: {len(df)}")
    print(f"output rows: {len(out_df)}")
    print(f"response column: {args.response_column}")
    print(f"skipped empty response: {skipped_empty}")
    print(f"skipped over max_length={args.max_length}: {skipped_long}")
    print(f"token length min/mean/p99/p999/max: {min(lengths)} / {sum(lengths)/len(lengths):.2f} / {p99} / {p999} / {max(lengths)}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
