#!/usr/bin/env python3
"""Decompose why top-k interaction mass saturates on Group-OPD prefixes."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


TOP_K = 16


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    # JSON strings may legally contain Unicode line separators such as U+2028.
    # JSONL records are separated only by the physical LF byte written by the
    # producer, not by every character recognized by str.splitlines().
    lines = raw.split("\n")
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A process interrupted during its final write can leave one
            # unterminated record. Earlier complete records remain valid.
            if index == len(lines) - 1 and not raw.endswith("\n"):
                print(f"Skipping unterminated final JSONL record in {path}", flush=True)
                continue
            raise
    return records


def normalize_prompt(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return [dict(message) for message in prompt]


def token_category(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "whitespace"
    if stripped.startswith("\\"):
        return "latex_control"
    if stripped.isdigit():
        return "number"
    if not any(character.isalnum() for character in stripped):
        return "punctuation"
    return "text_or_math"


def worker(args: tuple[Any, ...]) -> str:
    worker_id, gpu_id, input_path, student_model, teacher_model, states, batch_size, output_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frame = pd.read_parquet(input_path)
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = []
    for state in states:
        prompt = normalize_prompt(frame.iloc[state["source_row"]].to_dict()["prompt"])
        prepared.append(
            {
                **state,
                "base_ids": tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=True),
            }
        )
    student = teacher = None
    try:
        common = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
        student = AutoModelForCausalLM.from_pretrained(student_model, **common).eval().cuda()
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model, **common).eval().cuda()
        with open(output_path, "w", encoding="utf-8") as handle:
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                sequences = [item["base_ids"] + item["token_ids"] for item in batch]
                padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
                input_ids = padded["input_ids"].cuda()
                attention_mask = padded["attention_mask"].cuda()
                starts = [len(item["base_ids"]) for item in batch]
                ends = [len(sequence) for sequence in sequences]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    s_logits = student(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                    t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                for row, item in enumerate(batch):
                    s = s_logits[row, starts[row] - 1 : ends[row] - 1].float()
                    t = t_logits[row, starts[row] - 1 : ends[row] - 1].float()
                    s_top_logits, s_ids = torch.topk(s, k=TOP_K, dim=-1)
                    t_top_logits, t_ids = torch.topk(t, k=TOP_K, dim=-1)
                    s_probs = (s_top_logits - torch.logsumexp(s, dim=-1, keepdim=True)).exp()
                    t_probs = (t_top_logits - torch.logsumexp(t, dim=-1, keepdim=True)).exp()
                    eq = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2))
                    s_shared = eq.any(dim=-1)
                    t_shared = eq.any(dim=-2)
                    s_top_mass = s_probs.sum(dim=-1)
                    t_top_mass = t_probs.sum(dim=-1)
                    s_shared_mass = (s_probs * s_shared).sum(dim=-1)
                    t_shared_mass = (t_probs * t_shared).sum(dim=-1)
                    interaction = torch.minimum(s_shared_mass, t_shared_mass)
                    token_records = []
                    for position, token_id in enumerate(item["token_ids"]):
                        token_records.append(
                            {
                                "position": position,
                                "category": token_category(tokenizer.decode([token_id], skip_special_tokens=False)),
                                "student_top16_mass": float(s_top_mass[position]),
                                "teacher_top16_mass": float(t_top_mass[position]),
                                "intersection_count": int(s_shared[position].sum()),
                                "student_shared_mass": float(s_shared_mass[position]),
                                "teacher_shared_mass": float(t_shared_mass[position]),
                                "interaction_mass": float(interaction[position]),
                            }
                        )
                    handle.write(json.dumps({"source_row": item["source_row"], "candidate_id": item["candidate_id"], "tokens": token_records}) + "\n")
                del s_logits, t_logits
                print(f"[worker {worker_id}, gpu {gpu_id}] {min(start + len(batch), len(prepared))}/{len(prepared)} states", flush=True)
    finally:
        del student, teacher
        gc.collect()
        torch.cuda.empty_cache()
    return output_path


def stats(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    if not values:
        return {}
    return {
        "mean": sum(values) / len(values),
        "p01": values[int(.01 * (len(values) - 1))],
        "p10": values[int(.10 * (len(values) - 1))],
        "p50": values[int(.50 * (len(values) - 1))],
        "p90": values[int(.90 * (len(values) - 1))],
        "p99": values[int(.99 * (len(values) - 1))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = [record for path in sorted(Path(args.candidate_dir).glob("candidates_worker_*.jsonl")) for record in read_jsonl(path)]
    random.Random(args.seed).shuffle(records)
    records = records[: args.num_prompts]
    states = [
        {"source_row": record["source_row"], "candidate_id": candidate["candidate_id"], "token_ids": candidate["token_ids"]}
        for record in records
        for candidate in record["candidates"]
        if candidate["valid"]
    ]
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    shards = [[] for _ in gpu_ids]
    for index, state in enumerate(states):
        shards[index % len(shards)].append(state)
    context = multiprocessing.get_context("spawn")
    worker_args = [
        (index, gpu, args.input, args.student_model, args.teacher_model, shard, args.batch_size, str(output_dir / f"tokens_worker_{index:02d}.jsonl"))
        for index, (gpu, shard) in enumerate(zip(gpu_ids, shards))
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        paths = list(executor.map(worker, worker_args))
    token_rows = [token for path in paths for state in read_jsonl(Path(path)) for token in state["tokens"]]
    fields = ["student_top16_mass", "teacher_top16_mass", "intersection_count", "student_shared_mass", "teacher_shared_mass", "interaction_mass"]
    by_position = defaultdict(lambda: defaultdict(list))
    by_category = defaultdict(lambda: defaultdict(list))
    for token in token_rows:
        for field in fields:
            by_position[token["position"]][field].append(token[field])
            by_category[token["category"]][field].append(token[field])
    summary = {
        "num_prompts": len(records),
        "num_valid_states": len(states),
        "num_token_positions": len(token_rows),
        "overall": {field: stats([token[field] for token in token_rows]) for field in fields},
        "by_category": {category: {field: stats(values) for field, values in groups.items()} for category, groups in by_category.items()},
        "by_position": {str(position): {field: stats(values) for field, values in groups.items()} for position, groups in by_position.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
