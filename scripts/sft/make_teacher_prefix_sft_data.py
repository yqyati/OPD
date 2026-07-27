#!/usr/bin/env python3
"""Build pure-SFT data from generated teacher-response parquet.

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
    parser.add_argument("--chat-template-file", default=None)
    parser.add_argument("--use-generated-token-ids", action="store_true")
    parser.add_argument("--generated-token-ids-column", default="teacher_prefix_token_ids")
    parser.add_argument("--finish-reason-column", default="teacher_prefix_finish_reason")
    parser.add_argument("--source-eos-token-id", type=int, default=None)
    parser.add_argument("--canonical-eos-token-id", type=int, default=None)
    return parser.parse_args()


def normalize_prompt(prompt) -> list[dict]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"Expected non-empty prompt list, got {type(prompt)}")
    return [dict(m) for m in prompt]


def main() -> None:
    args = parse_args()
    if (args.source_eos_token_id is None) != (args.canonical_eos_token_id is None):
        raise ValueError("--source-eos-token-id and --canonical-eos-token-id must be set together")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    chat_template = None
    if args.chat_template_file is not None:
        chat_template = Path(args.chat_template_file).read_text(encoding="utf-8")
    df = pd.read_parquet(args.input)

    rows = []
    lengths = []
    skipped_empty = 0
    skipped_long = 0
    for _, row in df.iterrows():
        teacher_response = row.get(args.response_column, "")
        generated_ids = row.get(args.generated_token_ids_column)
        if args.use_generated_token_ids:
            if generated_ids is None:
                raise RuntimeError(
                    f"{args.generated_token_ids_column} is required with --use-generated-token-ids"
                )
            generated_ids = [int(token_id) for token_id in generated_ids]
            if row.get(args.finish_reason_column) == "stop" and args.source_eos_token_id is not None:
                if not generated_ids or generated_ids[-1] != args.source_eos_token_id:
                    raise RuntimeError(
                        f"Stopped response must end with source EOS {args.source_eos_token_id}; "
                        f"got {generated_ids[-1] if generated_ids else None}."
                    )
                generated_ids[-1] = args.canonical_eos_token_id
        if args.use_generated_token_ids and not generated_ids:
            skipped_empty += 1
            continue
        if not args.use_generated_token_ids and (teacher_response is None or str(teacher_response) == ""):
            skipped_empty += 1
            continue

        messages = normalize_prompt(row["prompt"])
        if args.use_generated_token_ids:
            base_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
                chat_template=chat_template,
            )
            base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
            token_ids = base_ids + generated_ids
            loss_mask = [0] * len(base_ids) + [1] * len(generated_ids)
            messages.append({"role": "assistant", "content": tokenizer.decode(generated_ids)})
        else:
            messages.append({"role": "assistant", "content": str(teacher_response)})
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=args.enable_thinking,
                chat_template=chat_template,
            )
            loss_mask = None
        length = len(token_ids)
        if length > args.max_length:
            skipped_long += 1
            continue

        out_row = row.to_dict()
        out_row["messages"] = messages
        out_row["enable_thinking"] = bool(args.enable_thinking)
        out_row["sft_token_len"] = length
        if args.use_generated_token_ids:
            out_row["precomputed_input_ids"] = token_ids
            out_row["precomputed_loss_mask"] = loss_mask
            out_row["sft_uses_generated_token_ids"] = True
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
    print(f"uses generated token ids: {args.use_generated_token_ids}")
    if args.use_generated_token_ids:
        print(f"generated token ids column: {args.generated_token_ids_column}")
    print(f"skipped empty response: {skipped_empty}")
    print(f"skipped over max_length={args.max_length}: {skipped_long}")
    print(f"token length min/mean/p99/p999/max: {min(lengths)} / {sum(lengths)/len(lengths):.2f} / {p99} / {p999} / {max(lengths)}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
