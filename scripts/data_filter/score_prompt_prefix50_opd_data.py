#!/usr/bin/env python3
"""Score OPD data with first-50-token rollout diagnostics as the main signal."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from score_prompt_opd_data import extract_prompt_text, get_torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original training parquet.")
    parser.add_argument("--prompt-scores", required=True, help="Existing prompt score parquet.")
    parser.add_argument("--prefix-rollouts", required=True, help="Parquet from generate_prefix50_rollouts.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--top-fracs", default="0.5,0.3")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true")
    return parser.parse_args()


def extract_prompt_chat(row: pd.Series):
    prompt = row.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)) and prompt:
        return list(prompt)
    return [{"role": "user", "content": extract_prompt_text(row)}]


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


def rank01(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True).fillna(0.0)


def positive_rank01(series: pd.Series) -> pd.Series:
    ranked = pd.Series(0.0, index=series.index)
    positive = series > 0
    ranked.loc[positive] = series.loc[positive].rank(method="average", pct=True)
    return ranked


def add_prefix50_scores(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    teacher_adv = df["prefix50_nll_gap"]
    teacher_conf = -df["prefix50_nll_teacher"]
    entropy_adv = df["prefix50_entropy_gap"]
    overlap = df[f"prefix50_top{topk}_overlap"].clip(lower=0)
    prompt_score = df["opd_prompt_score"] if "opd_prompt_score" in df else pd.Series(0.0, index=df.index)

    df["prefix50_teacher_adv_rank"] = positive_rank01(teacher_adv)
    df["prefix50_teacher_conf_rank"] = rank01(teacher_conf)
    df["prefix50_entropy_adv_rank"] = positive_rank01(entropy_adv)
    df["prefix50_topk_overlap_rank"] = rank01(overlap)
    df["prompt_score_rank"] = rank01(prompt_score)
    df["opd_prefix50_gap_score"] = (
        0.55 * df["prefix50_teacher_adv_rank"]
        + 0.30 * df["prefix50_teacher_conf_rank"]
        + 0.10 * df["prefix50_entropy_adv_rank"]
        + 0.05 * df["prompt_score_rank"]
    )
    df["opd_prefix50_score"] = (
        0.55 * df["prefix50_teacher_conf_rank"]
        + 0.25 * df["prefix50_topk_overlap_rank"]
        + 0.15 * df["prefix50_entropy_adv_rank"]
        + 0.05 * df["prompt_score_rank"]
    )
    df["opd_prefix50_agree_score"] = df["opd_prefix50_score"] * overlap
    return df


def write_subsets(df: pd.DataFrame, output_dir: Path, top_fracs: list[float]) -> None:
    sorted_df = df.sort_values("opd_prefix50_score", ascending=False)
    for frac in top_fracs:
        n = max(1, int(len(sorted_df) * frac))
        out_path = output_dir / f"opd_prefix50_score_top{int(frac * 100)}.parquet"
        sorted_df.head(n).to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({n} rows)")


def batch_metrics(
    formatted_prompts: list[str],
    prefixes: list[str],
    tokenizer,
    student,
    teacher,
    student_device: str,
    teacher_device: str,
    max_length: int,
    topk: int,
) -> list[dict[str, float]]:
    prompt_lens = [
        len(tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"])
        for prompt in formatted_prompts
    ]
    full_texts = [prompt + prefix for prompt, prefix in zip(formatted_prompts, prefixes)]
    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    target_ids = input_ids[:, 1:]
    selected_mask = torch.zeros_like(target_ids, dtype=torch.bool)
    for row_idx, prompt_len in enumerate(prompt_lens):
        valid_len = int(attention_mask[row_idx].sum().item())
        start_col = max(0, prompt_len - 1)
        end_col = max(start_col, valid_len - 1)
        selected_mask[row_idx, start_col:end_col] = True

    def collect(model, device: str):
        with torch.no_grad():
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits[:, :-1, :].float()
        selected_logits = logits[selected_mask.to(device)]
        selected_targets = target_ids[selected_mask].to(device)
        log_probs = F.log_softmax(selected_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).cpu()
        nll = -log_probs.gather(-1, selected_targets.unsqueeze(-1)).squeeze(-1).cpu()
        topk_ids = torch.topk(selected_logits, k=topk, dim=-1).indices.cpu()
        del logits
        return entropy, nll, topk_ids

    sample_indices = torch.nonzero(selected_mask, as_tuple=True)[0]
    student_entropy, student_nll, student_topk = collect(student, student_device)
    teacher_entropy, teacher_nll, teacher_topk = collect(teacher, teacher_device)
    overlap = (student_topk.unsqueeze(-1) == teacher_topk.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)

    rows = []
    for idx in range(input_ids.size(0)):
        token_mask = sample_indices == idx
        if not token_mask.any():
            rows.append(
                {
                    "prefix50_score_num_tokens": 0.0,
                    "prefix50_entropy_student": float("nan"),
                    "prefix50_entropy_teacher": float("nan"),
                    "prefix50_entropy_gap": float("nan"),
                    "prefix50_nll_student": float("nan"),
                    "prefix50_nll_teacher": float("nan"),
                    "prefix50_nll_gap": float("nan"),
                    f"prefix50_top{topk}_overlap": float("nan"),
                }
            )
            continue
        hs = student_entropy[token_mask]
        ht = teacher_entropy[token_mask]
        ns = student_nll[token_mask]
        nt = teacher_nll[token_mask]
        ov = overlap[token_mask]
        rows.append(
            {
                "prefix50_score_num_tokens": float(token_mask.sum().item()),
                "prefix50_entropy_student": float(hs.mean().item()),
                "prefix50_entropy_teacher": float(ht.mean().item()),
                "prefix50_entropy_gap": float((hs - ht).mean().item()),
                "prefix50_nll_student": float(ns.mean().item()),
                "prefix50_nll_teacher": float(nt.mean().item()),
                "prefix50_nll_gap": float((ns - nt).mean().item()),
                f"prefix50_top{topk}_overlap": float(ov.mean().item()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input).reset_index(drop=True)
    df["__opd_original_index"] = range(len(df))
    prompt_scores = pd.read_parquet(args.prompt_scores)
    prefix = pd.read_parquet(args.prefix_rollouts)
    keep_cols = [
        "__opd_original_index",
        "prompt_score_num_tokens",
        "prompt_entropy_student",
        "prompt_entropy_teacher",
        "prompt_entropy_gap",
        "prompt_nll_student",
        "prompt_nll_teacher",
        "prompt_nll_gap",
        f"prompt_top{args.topk}_overlap",
        "opd_prompt_score",
    ]
    prompt_scores = prompt_scores[[c for c in keep_cols if c in prompt_scores.columns]]
    df = df.merge(prompt_scores, on="__opd_original_index", how="left")
    df = df.merge(prefix[["__opd_original_index", "prefix50_text"]], on="__opd_original_index", how="inner")
    if len(df) != len(prefix):
        raise RuntimeError(f"Merged {len(df)} rows but prefix file has {len(prefix)} rows.")

    dtype = get_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab() and not args.allow_tokenizer_mismatch:
        raise ValueError("Student and teacher tokenizers differ. Use --allow-tokenizer-mismatch only if this is expected.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = load_model(args.student, args.student_device, dtype)
    teacher = load_model(args.teacher, args.teacher_device, dtype)

    formatted_prompts = [
        tokenizer.apply_chat_template(
            extract_prompt_chat(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for _, row in df.iterrows()
    ]
    prefixes = df["prefix50_text"].fillna("").astype(str).tolist()

    metrics = []
    for start in tqdm(range(0, len(df), args.batch_size), desc="Scoring prefix50"):
        metrics.extend(
            batch_metrics(
                formatted_prompts[start : start + args.batch_size],
                prefixes[start : start + args.batch_size],
                tokenizer,
                student,
                teacher,
                args.student_device,
                args.teacher_device,
                args.max_length,
                args.topk,
            )
        )

    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    scored = add_prefix50_scores(scored, args.topk)
    scored_path = output_dir / "opd_prefix50_scores.parquet"
    scored.to_parquet(scored_path, index=False)
    print(f"Wrote {scored_path} ({len(scored)} rows)")

    top_fracs = [float(x) for x in args.top_fracs.split(",") if x.strip()]
    write_subsets(scored, output_dir, top_fracs)
    summary_cols = [
        "prefix50_nll_gap",
        "prefix50_nll_teacher",
        "prefix50_entropy_gap",
        f"prefix50_top{args.topk}_overlap",
        "opd_prompt_score",
        "opd_prefix50_score",
    ]
    print(scored[summary_cols].describe(percentiles=[0.1, 0.2, 0.5, 0.8, 0.9, 0.95]))


if __name__ == "__main__":
    main()
