#!/usr/bin/env python3
"""Add teacher prefix top-k soft targets to a teacher-prefix parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def normalize_prompt(prompt) -> list[dict]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"Expected non-empty prompt list, got {type(prompt)}")
    return [dict(m) for m in prompt]


def get_teacher_prefix_text(row: pd.Series) -> str:
    value = row.get("teacher_prefix_text", "")
    if value is None:
        return ""
    return str(value)


def main() -> None:
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    df = pd.read_parquet(args.input)
    topk_ids_col: list[list[list[int]]] = []
    topk_logp_col: list[list[list[float]]] = []
    skipped_empty = 0
    skipped_long = 0

    pending = []
    for row_idx, row in tqdm(list(df.iterrows()), desc="prepare prompts"):
        teacher_prefix = get_teacher_prefix_text(row)
        if not teacher_prefix:
            pending.append((row_idx, None))
            skipped_empty += 1
            continue

        messages = normalize_prompt(row["prompt"])
        base_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=args.enable_thinking,
        )
        raw_prompt = base_prompt + teacher_prefix
        base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
        start = len(base_ids) - 1
        end = len(full_ids) - 1
        if start >= end:
            pending.append((row_idx, None))
            skipped_empty += 1
            continue
        if len(full_ids) > args.max_length:
            pending.append((row_idx, None))
            skipped_long += 1
            continue
        pending.append((row_idx, {"full_ids": full_ids, "start": start, "end": end}))

    results: dict[int, tuple[list[list[int]], list[list[float]]]] = {}
    valid_items = [(idx, item) for idx, item in pending if item is not None]
    for batch_start in tqdm(range(0, len(valid_items), args.batch_size), desc="teacher forward"):
        batch = valid_items[batch_start : batch_start + args.batch_size]
        max_len = max(len(item["full_ids"]) for _, item in batch)
        input_ids = []
        attention_mask = []
        for _, item in batch:
            ids = item["full_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=model.device)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids_t, attention_mask=attention_mask_t, use_cache=False).logits

        for batch_idx, (row_idx, item) in enumerate(batch):
            prefix_logits = logits[batch_idx, item["start"] : item["end"], :]
            prefix_log_probs = torch.log_softmax(prefix_logits.float(), dim=-1)
            values, indices = torch.topk(prefix_log_probs, k=args.top_k, dim=-1)
            results[row_idx] = (indices.cpu().tolist(), values.cpu().tolist())

    for row_idx, _ in pending:
        if row_idx in results:
            ids, logp = results[row_idx]
        else:
            ids, logp = [], []
        topk_ids_col.append(ids)
        topk_logp_col.append(logp)

    out_df = df.copy()
    out_df["teacher_prefix_top_k_ids"] = topk_ids_col
    out_df["teacher_prefix_top_k_log_probs"] = topk_logp_col
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)

    print(f"input rows: {len(df)}")
    print(f"rows with soft targets: {len(results)}")
    print(f"skipped empty/no-prefix: {skipped_empty}")
    print(f"skipped over max_length={args.max_length}: {skipped_long}")
    print(f"top_k: {args.top_k}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
