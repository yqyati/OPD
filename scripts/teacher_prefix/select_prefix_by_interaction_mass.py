#!/usr/bin/env python3
"""Select teacher-prefix handoff length with smoothed interaction mass.

This script is an offline preprocessing step. It reads a parquet that already
contains ``teacher_prefix_text`` and writes a new parquet with that field
replaced by a shorter selected prefix. The original prefix is preserved in
``original_teacher_prefix_text``.

For each sample, the script teacher-forces ``prompt + teacher_prefix`` through
both the student and teacher. A single forward pass per model gives next-token
logits at every prefix position, so every handoff length from 1..N can be scored
without N separate forwards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "data_filter"))

from score_prompt_opd_data import extract_prompt_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet with teacher_prefix_text.")
    parser.add_argument("--output", required=True, help="Output parquet with selected teacher_prefix_text.")
    parser.add_argument("--student-model", required=True, help="Student model path.")
    parser.add_argument("--teacher-model", required=True, help="Teacher model path.")
    parser.add_argument("--max-prefix-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--smooth-window", type=int, default=8)
    parser.add_argument("--min-prefix-len", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument(
        "--selection-rule",
        choices=["threshold", "argmax"],
        default="threshold",
        help="'threshold' chooses the earliest sufficient prefix; 'argmax' chooses the maximum smoothed score.",
    )
    parser.add_argument(
        "--fallback",
        choices=["max", "argmax", "zero"],
        default="max",
        help=(
            "Fallback when no smoothed score reaches threshold: "
            "'max' keeps max-prefix length, 'argmax' chooses the best smoothed score, "
            "'zero' drops the prefix."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows for smoke tests.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--device",
        default=None,
        help="Load both models on this single device, e.g. cuda:0. Overrides --device-map.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output parquet.")
    parser.add_argument(
        "--stats-jsonl",
        default=None,
        help="Optional jsonl path with per-sample selected length and score summaries.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def extract_prompt_chat(row: pd.Series) -> list[dict[str, str]]:
    prompt = row.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)) and prompt:
        return list(prompt)
    return [{"role": "user", "content": extract_prompt_text(row)}]


def get_prefix_text(row: pd.Series) -> str:
    prefix = row.get("teacher_prefix_text", "")
    if prefix is None:
        return ""
    if isinstance(prefix, float) and math.isnan(prefix):
        return ""
    return str(prefix)


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    total = 0.0
    q: list[float] = []
    for value in values:
        q.append(value)
        total += value
        if len(q) > window:
            total -= q.pop(0)
        out.append(total / len(q))
    return out


def select_length(
    *,
    scores: list[float],
    smooth_window: int,
    min_prefix_len: int,
    threshold: float,
    fallback: str,
    selection_rule: str,
) -> tuple[int, float, list[float]]:
    if not scores:
        return 0, 0.0, []

    smoothed = moving_average(scores, smooth_window)
    min_idx = max(0, min_prefix_len - 1)
    if selection_rule == "argmax":
        if min_idx >= len(smoothed):
            min_idx = len(smoothed) - 1
        idx = max(range(min_idx, len(smoothed)), key=lambda i: smoothed[i])
        return idx + 1, smoothed[idx], smoothed

    for idx in range(min_idx, len(smoothed)):
        if smoothed[idx] >= threshold:
            return idx + 1, smoothed[idx], smoothed

    if fallback == "zero":
        return 0, 0.0, smoothed
    if fallback == "argmax":
        idx = max(range(len(smoothed)), key=lambda i: smoothed[i])
        return idx + 1, smoothed[idx], smoothed
    idx = len(smoothed) - 1
    return idx + 1, smoothed[idx], smoothed


def assert_compatible_tokenizers(student_tok: Any, teacher_tok: Any) -> None:
    checks = [
        "hello",
        " Please reason step by step.",
        "\\boxed{42}",
        "因此我们有",
    ]
    for text in checks:
        student_ids = student_tok.encode(text, add_special_tokens=False)
        teacher_ids = teacher_tok.encode(text, add_special_tokens=False)
        if student_ids != teacher_ids:
            raise RuntimeError(
                "Student and teacher tokenizers are not compatible for token-id intersection. "
                f"Mismatch on text={text!r}: {student_ids[:20]} != {teacher_ids[:20]}"
            )


def build_item(row: pd.Series, tokenizer: Any, max_prefix_tokens: int, enable_thinking: bool) -> dict[str, Any]:
    base_text = tokenizer.apply_chat_template(
        extract_prompt_chat(row),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    prefix_text = get_prefix_text(row)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)[:max_prefix_tokens]
    full_ids = base_ids + prefix_ids
    return {
        "base_text": base_text,
        "base_len": len(base_ids),
        "prefix_ids": prefix_ids,
        "full_ids": full_ids,
    }


def pad_batch(items: list[dict[str, Any]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(item["full_ids"]) for item in items)
    input_ids = []
    attention_mask = []
    for item in items:
        ids = item["full_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def interaction_scores_for_item(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    base_len: int,
    prefix_len: int,
    top_k: int,
) -> list[float]:
    if prefix_len <= 0:
        return []

    positions = torch.arange(base_len, base_len + prefix_len, device=student_logits.device)
    s_logits = student_logits.index_select(0, positions).float()
    t_logits = teacher_logits.index_select(0, positions.to(teacher_logits.device)).float()

    s_logp = torch.log_softmax(s_logits, dim=-1)
    t_logp = torch.log_softmax(t_logits, dim=-1)
    s_vals, s_ids = torch.topk(s_logp, k=top_k, dim=-1)
    t_vals, t_ids = torch.topk(t_logp, k=top_k, dim=-1)

    if t_ids.device != s_ids.device:
        t_ids = t_ids.to(s_ids.device)
        t_vals = t_vals.to(s_vals.device)

    # eq[p, i, j] says whether student top-k token i equals teacher top-k token j
    # at prefix position p. Mass is summed only on intersected top-k tokens.
    eq = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2))
    s_shared = eq.any(dim=-1)
    t_shared = eq.any(dim=-2)

    student_mass = (s_vals.exp() * s_shared).sum(dim=-1)
    teacher_mass = (t_vals.exp() * t_shared).sum(dim=-1)
    scores = torch.minimum(student_mass, teacher_mass)
    return [float(x) for x in scores.detach().cpu().tolist()]


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"{output} already exists. Use --force to overwrite.")
        return

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    df = pd.read_parquet(args.input).reset_index(drop=True)
    df["__prefix_select_original_index"] = range(len(df))
    if args.num_shards > 1:
        df = df.iloc[args.shard_index :: args.num_shards].reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    student_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    assert_compatible_tokenizers(student_tok, teacher_tok)
    if student_tok.pad_token_id is None:
        student_tok.pad_token = student_tok.eos_token

    dtype = torch_dtype(args.dtype)
    print(f"Loading student: {args.student_model}")
    if args.device:
        student = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(args.device).eval()
    else:
        student = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
        ).eval()
    print(f"Loading teacher: {args.teacher_model}")
    if args.device:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(args.device).eval()
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
        ).eval()

    stats_path = Path(args.stats_jsonl) if args.stats_jsonl else output.with_suffix(".stats.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, Any]] = []
    with stats_path.open("w", encoding="utf-8") as stats_f:
        for start in range(0, len(df), args.batch_size):
            batch_df = df.iloc[start : start + args.batch_size]
            items = [
                build_item(row, student_tok, args.max_prefix_tokens, args.enable_thinking)
                for _, row in batch_df.iterrows()
            ]
            input_ids, attention_mask = pad_batch(items, student_tok.pad_token_id)

            with torch.inference_mode():
                student_out = student(
                    input_ids=input_ids.to(model_device(student)),
                    attention_mask=attention_mask.to(model_device(student)),
                    use_cache=False,
                )
                teacher_out = teacher(
                    input_ids=input_ids.to(model_device(teacher)),
                    attention_mask=attention_mask.to(model_device(teacher)),
                    use_cache=False,
                )

            # Keep logits on GPU. Moving full [batch, seq, vocab] logits to CPU
            # dominates runtime; only small top-k values are materialized on CPU.
            student_logits = student_out.logits.detach()
            teacher_logits = teacher_out.logits.detach()

            for local_idx, (_, row) in enumerate(batch_df.iterrows()):
                row_dict = row.to_dict()
                item = items[local_idx]
                prefix_len = len(item["prefix_ids"])
                scores = interaction_scores_for_item(
                    student_logits=student_logits[local_idx],
                    teacher_logits=teacher_logits[local_idx],
                    base_len=item["base_len"],
                    prefix_len=prefix_len,
                    top_k=args.top_k,
                )
                selected_len, selected_score, smoothed = select_length(
                    scores=scores,
                    smooth_window=args.smooth_window,
                    min_prefix_len=args.min_prefix_len,
                    threshold=args.threshold,
                    fallback=args.fallback,
                    selection_rule=args.selection_rule,
                )

                selected_prefix_ids = item["prefix_ids"][:selected_len]
                selected_prefix_text = student_tok.decode(selected_prefix_ids, skip_special_tokens=False)

                row_dict["original_teacher_prefix_text"] = get_prefix_text(row)
                row_dict["original_teacher_prefix_token_len"] = prefix_len
                row_dict["teacher_prefix_text"] = selected_prefix_text
                row_dict["selected_prefix_len"] = selected_len
                row_dict["selected_prefix_score"] = selected_score
                row_dict["selected_prefix_score_raw"] = scores[selected_len - 1] if selected_len > 0 else 0.0
                row_dict["selected_prefix_score_at_max"] = smoothed[-1] if smoothed else 0.0
                row_dict["selected_prefix_score_max"] = max(smoothed) if smoothed else 0.0
                row_dict["selected_prefix_score_mean"] = sum(smoothed) / len(smoothed) if smoothed else 0.0
                row_dict["selected_prefix_top_k"] = args.top_k
                row_dict["selected_prefix_smooth_window"] = args.smooth_window
                row_dict["selected_prefix_threshold"] = args.threshold
                row_dict["selected_prefix_selection_rule"] = args.selection_rule
                row_dict["selected_prefix_fallback"] = args.fallback
                out_rows.append(row_dict)

                stats = {
                    "row_index": int(start + local_idx),
                    "original_prefix_len": prefix_len,
                    "selected_prefix_len": selected_len,
                    "selected_prefix_score": selected_score,
                    "score_at_max": smoothed[-1] if smoothed else 0.0,
                    "score_max": max(smoothed) if smoothed else 0.0,
                    "score_mean": sum(smoothed) / len(smoothed) if smoothed else 0.0,
                    "score_at_32": smoothed[31] if len(smoothed) >= 32 else None,
                    "score_at_64": smoothed[63] if len(smoothed) >= 64 else None,
                    "score_at_96": smoothed[95] if len(smoothed) >= 96 else None,
                    "score_at_128": smoothed[127] if len(smoothed) >= 128 else None,
                }
                stats_f.write(json.dumps(stats, ensure_ascii=False) + "\n")

            done = min(start + len(batch_df), len(df))
            print(f"Processed {done}/{len(df)}")
            del student_out, teacher_out, student_logits, teacher_logits

    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(output, index=False)
    print(f"Wrote {output} ({len(out_df)} rows)")
    print(f"Wrote stats {stats_path}")


if __name__ == "__main__":
    main()
