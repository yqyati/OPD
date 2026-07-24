#!/usr/bin/env python3
"""Measure teacher signal remaining on short on-policy suffix probes.

Uses existing teacher-prefix handoff rollouts. For each saved student suffix,
the first probe_tokens are teacher-forced into both models after the exact
teacher prefix. No new generation or reward label is used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


TOP_K = 16


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def messages(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return [dict(item) for item in prompt]


def worker(args: tuple[Any, ...]) -> str:
    worker_id, gpu_id, input_path, student_model, teacher_model, records, probe_tokens, batch_size, output_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frame = pd.read_parquet(input_path)
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = []
    for record in records:
        row = frame.iloc[record["source_row"]].to_dict()
        base_ids = tokenizer.apply_chat_template(messages(row["prompt"]), tokenize=True, add_generation_prompt=True, enable_thinking=True)
        teacher_prefix = [int(token) for token in row["teacher_prefix_token_ids"][: record["requested_prefix_len"]]]
        response = [int(token) for token in record["response_token_ids"][:probe_tokens]]
        if not response:
            continue
        prepared.append({**record, "base_ids": base_ids, "teacher_prefix": teacher_prefix, "probe_ids": response})

    student = teacher = None
    try:
        common = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
        student = AutoModelForCausalLM.from_pretrained(student_model, **common).eval().cuda()
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model, **common).eval().cuda()
        with open(output_path, "w", encoding="utf-8") as handle:
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                sequences = [item["base_ids"] + item["teacher_prefix"] + item["probe_ids"] for item in batch]
                padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
                input_ids = padded["input_ids"].cuda()
                attention_mask = padded["attention_mask"].cuda()
                suffix_starts = [len(item["base_ids"]) + len(item["teacher_prefix"]) for item in batch]
                suffix_ends = [start_index + len(item["probe_ids"]) for start_index, item in zip(suffix_starts, batch)]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    s_logits = student(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                    t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                for row, item in enumerate(batch):
                    start_index, end_index = suffix_starts[row], suffix_ends[row]
                    target_ids = input_ids[row, start_index:end_index]
                    s = s_logits[row, start_index - 1 : end_index - 1].float()
                    t = t_logits[row, start_index - 1 : end_index - 1].float()
                    s_logp = s - torch.logsumexp(s, dim=-1, keepdim=True)
                    t_logp = t - torch.logsumexp(t, dim=-1, keepdim=True)
                    t_top_logp, t_top_ids = torch.topk(t_logp, k=TOP_K, dim=-1)
                    t_top_p = t_top_logp.exp()
                    student_at_teacher_top = s_logp.gather(-1, t_top_ids)
                    top16_forward_kl = (t_top_p * (t_top_logp - student_at_teacher_top)).sum(dim=-1)
                    s_top_logits, s_top_ids = torch.topk(s, k=TOP_K, dim=-1)
                    s_top_p = (s_top_logits - torch.logsumexp(s, dim=-1, keepdim=True)).exp()
                    eq = s_top_ids.unsqueeze(-1).eq(t_top_ids.unsqueeze(-2))
                    interaction = torch.minimum(
                        (s_top_p * eq.any(dim=-1)).sum(dim=-1),
                        (t_top_p * eq.any(dim=-2)).sum(dim=-1),
                    )
                    actual_advantage = (
                        t_logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                        - s_logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                    )
                    midpoint = max(1, len(top16_forward_kl) // 2)
                    kl_slope = (
                        float(top16_forward_kl[midpoint:].mean() - top16_forward_kl[:midpoint].mean())
                        if len(top16_forward_kl) > 1
                        else None
                    )
                    handle.write(
                        json.dumps(
                            {
                                "source_row": item["source_row"],
                                "requested_prefix_len": item["requested_prefix_len"],
                                "continuation_id": item["continuation_id"],
                                "probe_length": len(item["probe_ids"]),
                                "rule_score": item["rule_score"],
                                "suffix_top16_forward_kl_mean": float(top16_forward_kl.mean()),
                                "suffix_teacher_advantage_mean": float(actual_advantage.mean()),
                                "suffix_interaction_mass_mean": float(interaction.mean()),
                                "suffix_kl_second_minus_first": kl_slope,
                            }
                        )
                        + "\n"
                    )
                del s_logits, t_logits
                print(f"[worker {worker_id}, gpu {gpu_id}] {min(start + len(batch), len(prepared))}/{len(prepared)} suffix probes", flush=True)
    finally:
        del student, teacher
        gc.collect()
        torch.cuda.empty_cache()
    return output_path


def summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[row["requested_prefix_len"]].append(row)
    fields = [
        "probe_length",
        "rule_score",
        "suffix_top16_forward_kl_mean",
        "suffix_teacher_advantage_mean",
        "suffix_interaction_mass_mean",
        "suffix_kl_second_minus_first",
    ]
    output = {}
    for length, group in sorted(by_length.items()):
        output[str(length)] = {}
        for field in fields:
            values = [row[field] for row in group if row[field] is not None]
            output[str(length)][field] = sum(values) / len(values) if values else None
        output[str(length)]["num_probes"] = len(group)
    (output_dir / "summary.json").write_text(json.dumps(output, indent=2))
    with open(output_dir / "suffix_probe_scores.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(output, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--handoff-rollouts", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-tokens", type=int, default=128)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = read_jsonl(Path(args.handoff_rollouts))
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    shards = [[] for _ in gpu_ids]
    for index, record in enumerate(records):
        shards[index % len(shards)].append(record)
    context = multiprocessing.get_context("spawn")
    worker_args = [
        (index, gpu, args.input, args.student_model, args.teacher_model, shard, args.probe_tokens, args.batch_size, str(output_dir / f"probe_worker_{index:02d}.jsonl"))
        for index, (gpu, shard) in enumerate(zip(gpu_ids, shards))
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        paths = list(executor.map(worker, worker_args))
    summary([row for path in paths for row in read_jsonl(Path(path))], output_dir)


if __name__ == "__main__":
    main()
