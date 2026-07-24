#!/usr/bin/env python3
"""Test whether early student excess entropy predicts teacher-prefix uplift.

This is a forward-only diagnostic.  It joins matched, completed student
rollouts from Prefix0 and Prefix128, then teacher-forces the first teacher
tokens through frozen student and teacher models.  No training code, rollout,
or teacher semantic judgement is involved.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import math
import multiprocessing
import os
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


def parse_windows(value: str) -> list[int]:
    windows = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not windows or windows[0] <= 0:
        raise argparse.ArgumentTypeError("--windows must contain positive integers")
    return windows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_messages(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return [dict(item) for item in prompt]


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(average_ranks(left), average_ranks(right))


def load_matched_values(prefix0_path: Path, prefix128_path: Path) -> list[dict[str, Any]]:
    zero = {int(row["source_row"]): row for row in read_jsonl(prefix0_path)}
    prefix = {int(row["source_row"]): row for row in read_jsonl(prefix128_path)}
    if set(zero) != set(prefix):
        missing_zero = sorted(set(prefix) - set(zero))[:10]
        missing_prefix = sorted(set(zero) - set(prefix))[:10]
        raise RuntimeError(f"Prefix0/Prefix128 source rows differ; missing_zero={missing_zero}, missing_prefix={missing_prefix}")
    joined = []
    for source_row in sorted(zero):
        v0 = float(zero[source_row]["continuation_value_by_prefix_len"]["0"])
        v128 = float(prefix[source_row]["continuation_value_by_prefix_len"]["128"])
        joined.append({"source_row": source_row, "V0": v0, "V128": v128, "prefix128_uplift": v128 - v0})
    return joined


def worker(args: tuple[Any, ...]) -> str:
    worker_id, gpu_id, input_path, student_model, teacher_model, samples, windows, batch_size, output_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    max_window = max(windows)
    frame = pd.read_parquet(input_path)
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prepared = []
    for item in samples:
        row = frame.iloc[item["source_row"]].to_dict()
        base_ids = tokenizer.apply_chat_template(
            prompt_messages(row["prompt"]), tokenize=True, add_generation_prompt=True, enable_thinking=True
        )
        prefix_ids = [int(token) for token in row["teacher_prefix_token_ids"]][:max_window]
        if len(prefix_ids) < max_window:
            raise RuntimeError(f"source_row={item['source_row']} has fewer than {max_window} teacher tokens")
        prepared.append({**item, "base_ids": base_ids, "prefix_ids": prefix_ids})

    student = teacher = None
    try:
        common = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
        student = AutoModelForCausalLM.from_pretrained(student_model, **common).eval().cuda()
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model, **common).eval().cuda()
        with open(output_path, "w", encoding="utf-8") as handle:
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                sequences = [item["base_ids"] + item["prefix_ids"] for item in batch]
                padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
                input_ids = padded["input_ids"].cuda()
                attention_mask = padded["attention_mask"].cuda()
                starts = [len(item["base_ids"]) for item in batch]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    student_logits = student(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                    teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                for row_index, item in enumerate(batch):
                    # Logit at start-1 predicts the first teacher token.  Every
                    # model sees the exact same teacher-forced history here.
                    student_slice = student_logits[row_index, starts[row_index] - 1 : starts[row_index] - 1 + max_window].float()
                    teacher_slice = teacher_logits[row_index, starts[row_index] - 1 : starts[row_index] - 1 + max_window].float()
                    student_logp = torch.log_softmax(student_slice, dim=-1)
                    teacher_logp = torch.log_softmax(teacher_slice, dim=-1)
                    student_prob = student_logp.exp()
                    teacher_prob = teacher_logp.exp()
                    student_entropy = -(student_prob * student_logp).sum(dim=-1)
                    teacher_entropy = -(teacher_prob * teacher_logp).sum(dim=-1)
                    # These are prompt-conditioned policy-disagreement metrics,
                    # not entropy comparisons.  They remain well defined when
                    # both models have equally sharp but different next-token
                    # distributions.
                    teacher_to_student_kl = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
                    mixture_logp = torch.logaddexp(student_logp, teacher_logp) - math.log(2.0)
                    js_divergence = 0.5 * (
                        (student_prob * (student_logp - mixture_logp)).sum(dim=-1)
                        + (teacher_prob * (teacher_logp - mixture_logp)).sum(dim=-1)
                    )
                    teacher_targets = input_ids[row_index, starts[row_index] : starts[row_index] + max_window]
                    student_teacher_token_nll = -student_logp.gather(-1, teacher_targets.unsqueeze(-1)).squeeze(-1)
                    record: dict[str, Any] = {
                        "source_row": item["source_row"],
                        "V0": item["V0"],
                        "V128": item["V128"],
                        "prefix128_uplift": item["prefix128_uplift"],
                    }
                    for window in windows:
                        student_mean = float(student_entropy[:window].mean())
                        teacher_mean = float(teacher_entropy[:window].mean())
                        record[f"student_entropy_w{window}"] = student_mean
                        record[f"teacher_entropy_w{window}"] = teacher_mean
                        record[f"excess_entropy_w{window}"] = student_mean - teacher_mean
                        record[f"teacher_to_student_kl_w{window}"] = float(teacher_to_student_kl[:window].mean())
                        record[f"js_divergence_w{window}"] = float(js_divergence[:window].mean())
                        record[f"student_teacher_token_nll_w{window}"] = float(student_teacher_token_nll[:window].mean())
                    handle.write(json.dumps(record) + "\n")
                del student_logits, teacher_logits
                print(f"[worker {worker_id}, gpu {gpu_id}] {min(start + len(batch), len(prepared))}/{len(prepared)} prompts", flush=True)
    finally:
        del student, teacher
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return output_path


def summarize(rows: list[dict[str, Any]], windows: list[int], output_dir: Path) -> None:
    uplift = [float(row["prefix128_uplift"]) for row in rows]
    summary: dict[str, Any] = {
        "num_prompts": len(rows),
        "mean_V0": mean(float(row["V0"]) for row in rows),
        "mean_V128": mean(float(row["V128"]) for row in rows),
        "mean_prefix128_uplift": mean(uplift),
        "fraction_positive_uplift": mean(value > 0.0 for value in uplift),
        "windows": {},
    }
    quartile_size = max(1, len(rows) // 4)
    for window in windows:
        window_summary: dict[str, Any] = {}
        for metric in (
            f"student_entropy_w{window}",
            f"teacher_entropy_w{window}",
            f"excess_entropy_w{window}",
            f"teacher_to_student_kl_w{window}",
            f"js_divergence_w{window}",
            f"student_teacher_token_nll_w{window}",
        ):
            scores = [float(row[metric]) for row in rows]
            ordered = sorted(range(len(rows)), key=lambda index: scores[index])
            bottom = ordered[:quartile_size]
            top = ordered[-quartile_size:]
            window_summary[metric] = {
                "mean": mean(scores),
                "pearson_with_prefix128_uplift": pearson(scores, uplift),
                "spearman_with_prefix128_uplift": spearman(scores, uplift),
                "bottom_quartile_mean_uplift": mean(uplift[index] for index in bottom),
                "top_quartile_mean_uplift": mean(uplift[index] for index in top),
                "top_minus_bottom_uplift": mean(uplift[index] for index in top) - mean(uplift[index] for index in bottom),
                "top_quartile_mean_V0": mean(float(rows[index]["V0"]) for index in top),
                "top_quartile_mean_V128": mean(float(rows[index]["V128"]) for index in top),
            }
        summary["windows"][str(window)] = window_summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output_dir / "entropy_uplift_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prefix0-values", required=True)
    parser.add_argument("--prefix128-values", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--windows", type=parse_windows, default=parse_windows("1,4,8,16,32"))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = load_matched_values(Path(args.prefix0_values), Path(args.prefix128_values))
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU")
    shards = [[] for _ in gpu_ids]
    for index, sample in enumerate(samples):
        shards[index % len(shards)].append(sample)
    context = multiprocessing.get_context("spawn")
    worker_args = [
        (index, gpu, args.input, args.student_model, args.teacher_model, shard, args.windows, args.batch_size, str(output_dir / f"entropy_worker_{index:02d}.jsonl"))
        for index, (gpu, shard) in enumerate(zip(gpu_ids, shards))
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        paths = list(executor.map(worker, worker_args))
    rows = [row for path in paths for row in read_jsonl(Path(path))]
    if len(rows) != len(samples):
        raise RuntimeError(f"expected {len(samples)} entropy rows, found {len(rows)}")
    summarize(rows, args.windows, output_dir)


if __name__ == "__main__":
    main()
