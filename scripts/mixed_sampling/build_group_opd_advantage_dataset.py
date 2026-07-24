#!/usr/bin/env python3
"""Select one Prefix128 per prompt by a frozen teacher/student trajectory metric."""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # File iteration splits only physical LF records, preserving U+2028 inside
    # generated text fields.
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_prompt(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return [dict(message) for message in prompt]


def worker(args: tuple[Any, ...]) -> str:
    worker_id, gpu_id, input_path, student_model, teacher_model, records, metric, batch_size, output_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frame = pd.read_parquet(input_path)
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    states = []
    for record in records:
        prompt = normalize_prompt(frame.iloc[record["source_row"]].to_dict()["prompt"])
        base_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        for candidate in record["candidates"]:
            if candidate["valid"]:
                states.append(
                    {
                        "source_row": record["source_row"],
                        "candidate_id": candidate["candidate_id"],
                        "prefix_token_ids": candidate["token_ids"],
                        "prefix_text": candidate["text"],
                        "base_ids": base_ids,
                    }
                )

    student = teacher = None
    scored: dict[int, list[dict[str, Any]]] = defaultdict(list)
    try:
        common = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
        student = AutoModelForCausalLM.from_pretrained(student_model, **common).eval().cuda()
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model, **common).eval().cuda()
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            sequences = [item["base_ids"] + item["prefix_token_ids"] for item in batch]
            padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
            input_ids = padded["input_ids"].cuda()
            attention_mask = padded["attention_mask"].cuda()
            starts = [len(item["base_ids"]) for item in batch]
            ends = [len(sequence) for sequence in sequences]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = student(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            for row, item in enumerate(batch):
                s_logits = student_logits[row, starts[row] - 1 : ends[row] - 1].float()
                t_logits = teacher_logits[row, starts[row] - 1 : ends[row] - 1].float()
                if metric == "trajectory_teacher_advantage":
                    target_ids = input_ids[row, starts[row] : ends[row]]
                    s_logp = s_logits - torch.logsumexp(s_logits, dim=-1, keepdim=True)
                    t_logp = t_logits - torch.logsumexp(t_logits, dim=-1, keepdim=True)
                    score = (t_logp.gather(-1, target_ids.unsqueeze(-1)) - s_logp.gather(-1, target_ids.unsqueeze(-1))).mean()
                elif metric == "mean_top16_overlap":
                    _, s_ids = torch.topk(s_logits, k=16, dim=-1)
                    _, t_ids = torch.topk(t_logits, k=16, dim=-1)
                    overlap = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
                    score = overlap.mean()
                elif metric == "tail32_interaction_mass":
                    # Score only the local state immediately before suffix
                    # rollout. This matches trajectory_interaction_tail_quarter
                    # from the Prefix128 offline state-value benchmark.
                    tail_start = max(0, s_logits.shape[0] - 32)
                    s_tail = s_logits[tail_start:]
                    t_tail = t_logits[tail_start:]
                    s_top_logits, s_ids = torch.topk(s_tail, k=16, dim=-1)
                    t_top_logits, t_ids = torch.topk(t_tail, k=16, dim=-1)
                    s_top_logp = s_top_logits - torch.logsumexp(s_tail, dim=-1, keepdim=True)
                    t_top_logp = t_top_logits - torch.logsumexp(t_tail, dim=-1, keepdim=True)
                    shared_student = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2)).any(dim=-1)
                    shared_teacher = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2)).any(dim=-2)
                    student_mass = (s_top_logp.exp() * shared_student).sum(dim=-1)
                    teacher_mass = (t_top_logp.exp() * shared_teacher).sum(dim=-1)
                    score = torch.minimum(student_mass, teacher_mass).mean()
                elif metric == "position_weighted_interaction_mass":
                    # A parameter-free full-trajectory score: preserve all
                    # Prefix128 evidence while linearly emphasizing the state
                    # nearest to suffix handoff.
                    s_top_logits, s_ids = torch.topk(s_logits, k=16, dim=-1)
                    t_top_logits, t_ids = torch.topk(t_logits, k=16, dim=-1)
                    s_top_logp = s_top_logits - torch.logsumexp(s_logits, dim=-1, keepdim=True)
                    t_top_logp = t_top_logits - torch.logsumexp(t_logits, dim=-1, keepdim=True)
                    shared_student = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2)).any(dim=-1)
                    shared_teacher = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2)).any(dim=-2)
                    student_mass = (s_top_logp.exp() * shared_student).sum(dim=-1)
                    teacher_mass = (t_top_logp.exp() * shared_teacher).sum(dim=-1)
                    interaction = torch.minimum(student_mass, teacher_mass)
                    weights = torch.arange(1, interaction.numel() + 1, dtype=interaction.dtype, device=interaction.device)
                    score = (interaction * weights).sum() / weights.sum()
                else:
                    raise ValueError(f"unsupported metric: {metric}")
                scored[item["source_row"]].append({**item, "selector_score": float(score)})
            del student_logits, teacher_logits
            print(f"[advantage worker {worker_id}, gpu {gpu_id}] {min(start + len(batch), len(states))}/{len(states)} candidates", flush=True)
    finally:
        del student, teacher
        gc.collect()
        torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as handle:
        for source_row, candidates in sorted(scored.items()):
            selected = max(candidates, key=lambda item: (item["selector_score"], -item["candidate_id"]))
            handle.write(json.dumps({**selected, "valid_candidate_count": len(candidates)}, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument(
        "--metric",
        choices=[
            "trajectory_teacher_advantage",
            "mean_top16_overlap",
            "tail32_interaction_mass",
            "position_weighted_interaction_mass",
        ],
        default="trajectory_teacher_advantage",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    records = [record for path in sorted(Path(args.candidate_dir).glob("candidates_worker_*.jsonl")) for record in read_jsonl(path)]
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    shards = [[] for _ in gpu_ids]
    for index, record in enumerate(records):
        shards[index % len(shards)].append(record)
    context = multiprocessing.get_context("spawn")
    worker_args = [
        (
            index,
            gpu,
            args.input,
            args.student_model,
            args.teacher_model,
            shard,
            args.metric,
            args.batch_size,
            str(output.parent / f"selector_{args.metric}_worker_{index:02d}.jsonl"),
        )
        for index, (gpu, shard) in enumerate(zip(gpu_ids, shards))
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        paths = list(executor.map(worker, worker_args))
    selected = [record for path in paths for record in read_jsonl(Path(path))]
    frame = pd.read_parquet(args.input)
    selected_by_row = {record["source_row"]: record for record in selected}
    rows = sorted(selected_by_row)
    result = frame.iloc[rows].copy().reset_index(drop=True)
    result["teacher_prefix_token_ids"] = [selected_by_row[row]["prefix_token_ids"] for row in rows]
    result["teacher_prefix_text"] = [selected_by_row[row]["prefix_text"] for row in rows]
    result["teacher_prefix_token_len"] = 128
    result["teacher_prefix_finish_reason"] = "length"
    result["teacher_prefix_model"] = args.student_model
    result["teacher_prefix_max_tokens"] = 128
    result["teacher_prefix_temperature"] = 1.0
    result["teacher_prefix_top_p"] = 1.0
    result["teacher_prefix_enable_thinking"] = True
    result["group_opd_selector_rule"] = f"highest_{args.metric}"
    result["group_opd_selector_score"] = [selected_by_row[row]["selector_score"] for row in rows]
    result["group_opd_candidate_id"] = [selected_by_row[row]["candidate_id"] for row in rows]
    result["group_opd_valid_candidate_count"] = [selected_by_row[row]["valid_candidate_count"] for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    summary = {
        "input_rows": len(frame),
        "selected_rows": len(rows),
        "dropped_no_valid_candidate": len(frame) - len(rows),
        "metric": args.metric,
        "selected_score_mean": sum(record["selector_score"] for record in selected) / len(selected),
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
