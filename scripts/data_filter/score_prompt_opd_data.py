#!/usr/bin/env python3
"""Score OPD training prompts with no-rollout student/teacher diagnostics.

This script scores each prompt using teacher-forced prompt-token statistics:
entropy gap, NLL gap, and top-k overlap. It then writes a scored parquet plus
simple filtered subsets for quick OPD data-selection experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TEACHER_ALIGNED_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input training parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory for scored/subset parquets.")
    parser.add_argument("--student", required=True, help="Student model path.")
    parser.add_argument("--teacher", required=True, help="Teacher model path.")
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--top-fracs", default="0.5,0.3", help="Comma-separated top score fractions.")
    parser.add_argument("--tail-fraction", type=float, default=0.7, help="Fraction of later question tokens to score.")
    parser.add_argument("--min-prefix-tokens", type=int, default=8, help="Ignore early target positions.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true")
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def extract_prompt_text(row: pd.Series) -> str:
    if "prompt" in row and row["prompt"] is not None:
        prompt = row["prompt"]
        if isinstance(prompt, list) and prompt:
            item = prompt[0]
            if isinstance(item, dict) and "content" in item:
                return str(item["content"]).strip()
        if isinstance(prompt, str):
            return prompt.strip()
    if "problem" in row and row["problem"] is not None:
        return str(row["problem"]).strip()
    if "question" in row and row["question"] is not None:
        return str(row["question"]).strip()
    raise ValueError("Cannot find prompt/problem/question column in row.")


def strip_instruction(prompt: str) -> str:
    prompt = prompt.strip()
    if prompt.endswith(TEACHER_ALIGNED_SUFFIX):
        return prompt[: -len(TEACHER_ALIGNED_SUFFIX)].strip()
    marker = " Please reason step by step,"
    if marker in prompt:
        return prompt.split(marker, 1)[0].strip()
    return prompt


def load_model(path: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    return model


def topk_overlap(student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int) -> torch.Tensor:
    student_topk = torch.topk(student_logits, k=k, dim=-1).indices
    teacher_topk = torch.topk(teacher_logits, k=k, dim=-1).indices
    return (student_topk.unsqueeze(-1) == teacher_topk.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)


def score_batch(
    texts: list[str],
    tokenizer,
    student_model,
    teacher_model,
    student_device: str,
    teacher_device: str,
    max_length: int,
    topk: int,
    tail_fraction: float,
    min_prefix_tokens: int,
) -> list[dict[str, float]]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        student_logits = student_model(
            input_ids=input_ids.to(student_device),
            attention_mask=attention_mask.to(student_device),
        ).logits[:, :-1].float().cpu()
        teacher_logits = teacher_model(
            input_ids=input_ids.to(teacher_device),
            attention_mask=attention_mask.to(teacher_device),
        ).logits[:, :-1].float().cpu()

    target_ids = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:].bool()

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()

    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
    teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    student_nll = -student_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    teacher_nll = -teacher_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    overlap = topk_overlap(student_logits, teacher_logits, topk)

    rows = []
    for idx in range(input_ids.size(0)):
        valid = torch.nonzero(target_mask[idx], as_tuple=False).flatten()
        if valid.numel() == 0:
            rows.append(empty_metrics())
            continue

        start = min_prefix_tokens
        tail_start = int(valid.numel() * (1.0 - tail_fraction))
        start = max(start, tail_start)
        selected = valid[valid >= start]
        if selected.numel() == 0:
            selected = valid

        hs = student_entropy[idx, selected]
        ht = teacher_entropy[idx, selected]
        ns = student_nll[idx, selected]
        nt = teacher_nll[idx, selected]
        ov = overlap[idx, selected]
        rows.append(
            {
                "prompt_score_num_tokens": float(selected.numel()),
                "prompt_entropy_student": float(hs.mean().item()),
                "prompt_entropy_teacher": float(ht.mean().item()),
                "prompt_entropy_gap": float((hs - ht).mean().item()),
                "prompt_nll_student": float(ns.mean().item()),
                "prompt_nll_teacher": float(nt.mean().item()),
                "prompt_nll_gap": float((ns - nt).mean().item()),
                f"prompt_top{topk}_overlap": float(ov.mean().item()),
            }
        )
    return rows


def empty_metrics() -> dict[str, float]:
    return {
        "prompt_score_num_tokens": 0.0,
        "prompt_entropy_student": float("nan"),
        "prompt_entropy_teacher": float("nan"),
        "prompt_entropy_gap": float("nan"),
        "prompt_nll_student": float("nan"),
        "prompt_nll_teacher": float("nan"),
        "prompt_nll_gap": float("nan"),
    }


def add_scores(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    entropy_gap = df["prompt_entropy_gap"].clip(lower=0)
    nll_gap = df["prompt_nll_gap"].clip(lower=0)
    entropy_clip = entropy_gap.quantile(0.95)
    nll_clip = nll_gap.quantile(0.95)
    overlap = df[f"prompt_top{topk}_overlap"].clip(lower=0)
    df["opd_prompt_score"] = overlap * entropy_gap.clip(upper=entropy_clip) * nll_gap.clip(upper=nll_clip)
    return df


def write_subsets(df: pd.DataFrame, output_dir: Path, top_fracs: list[float], topk: int) -> None:
    sorted_df = df.sort_values("opd_prompt_score", ascending=False)
    stem = "opd_prompt_score"

    for frac in top_fracs:
        n = max(1, int(len(sorted_df) * frac))
        out_path = output_dir / f"{stem}_top{int(frac * 100)}.parquet"
        sorted_df.head(n).to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({n} rows)")

    overlap_q20 = df[f"prompt_top{topk}_overlap"].quantile(0.20)
    entropy_q20 = df["prompt_entropy_gap"].quantile(0.20)
    entropy_q95 = df["prompt_entropy_gap"].quantile(0.95)
    nll_q20 = df["prompt_nll_gap"].quantile(0.20)
    mid = df[
        (df[f"prompt_top{topk}_overlap"] >= overlap_q20)
        & (df["prompt_entropy_gap"] >= entropy_q20)
        & (df["prompt_entropy_gap"] <= entropy_q95)
        & (df["prompt_nll_gap"] >= nll_q20)
    ].sort_values("opd_prompt_score", ascending=False)
    out_path = output_dir / f"{stem}_mid.parquet"
    mid.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(mid)} rows)")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    prompts = [extract_prompt_text(row) for _, row in df.iterrows()]
    question_texts = [strip_instruction(prompt) for prompt in prompts]

    dtype = get_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab() and not args.allow_tokenizer_mismatch:
        raise ValueError("Student and teacher tokenizers differ. Use --allow-tokenizer-mismatch only if this is expected.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = load_model(args.student, args.student_device, dtype)
    teacher = load_model(args.teacher, args.teacher_device, dtype)

    metrics = []
    for start in tqdm(range(0, len(question_texts), args.batch_size), desc="Scoring prompts"):
        batch_texts = question_texts[start : start + args.batch_size]
        metrics.extend(
            score_batch(
                batch_texts,
                tokenizer,
                student,
                teacher,
                args.student_device,
                args.teacher_device,
                args.max_length,
                args.topk,
                args.tail_fraction,
                args.min_prefix_tokens,
            )
        )

    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    scored = add_scores(scored, args.topk)

    scored_path = output_dir / "opd_prompt_scores.parquet"
    scored.to_parquet(scored_path, index=False)
    print(f"Wrote {scored_path} ({len(scored)} rows)")

    top_fracs = [float(x) for x in args.top_fracs.split(",") if x.strip()]
    write_subsets(scored, output_dir, top_fracs, args.topk)

    summary_cols = [
        "prompt_entropy_gap",
        "prompt_nll_gap",
        f"prompt_top{args.topk}_overlap",
        "opd_prompt_score",
    ]
    print(scored[summary_cols].describe(percentiles=[0.1, 0.2, 0.5, 0.8, 0.9, 0.95]))


if __name__ == "__main__":
    main()
