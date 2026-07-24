#!/usr/bin/env python3
"""Build matched High-Median and Random-State Group-OPD training datasets.

The script samples one shared pool of student Prefix128 candidates per training
prompt.  It then ranks valid candidates with frozen teacher/student forwards
and writes two parquets differing only in the selected candidate state.
"""

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


def normalize_prompt(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"expected a non-empty chat prompt, got {type(prompt)}")
    return [dict(message) for message in prompt]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate_worker(args: tuple[Any, ...]) -> str:
    worker_id, gpu_id, model_path, samples, candidates_per_prompt, prefix_tokens, temperature, top_p, batch_size, output_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    llm = None
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=2304,
        )
        tokenizer = llm.get_tokenizer()
        prompt_ids = [
            tokenizer.apply_chat_template(
                sample["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=True
            )
            for sample in samples
        ]
        params = SamplingParams(
            n=candidates_per_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=prefix_tokens,
        )
        eos_id = tokenizer.eos_token_id
        with open(output_path, "w", encoding="utf-8") as handle:
            for start in range(0, len(samples), batch_size):
                batch_samples = samples[start : start + batch_size]
                outputs = llm.generate(
                    [{"prompt_token_ids": ids} for ids in prompt_ids[start : start + batch_size]], params, use_tqdm=False
                )
                for sample, output in zip(batch_samples, outputs):
                    candidates = []
                    for candidate_id, candidate in enumerate(output.outputs):
                        token_ids = [int(token_id) for token_id in candidate.token_ids]
                        ended = candidate.finish_reason == "stop" or (eos_id is not None and eos_id in token_ids)
                        candidates.append(
                            {
                                "candidate_id": candidate_id,
                                "token_ids": token_ids,
                                "text": candidate.text,
                                "finish_reason": candidate.finish_reason,
                                "valid": bool(not ended and len(token_ids) == prefix_tokens),
                            }
                        )
                    handle.write(json.dumps({"source_row": sample["source_row"], "candidates": candidates}, ensure_ascii=False) + "\n")
                print(f"[generate worker {worker_id}, gpu {gpu_id}] {min(start + len(batch_samples), len(samples))}/{len(samples)} prompts", flush=True)
    finally:
        if llm is not None:
            del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return output_path


def median_interaction_scores(student_logits: Any, teacher_logits: Any, starts: list[int], ends: list[int]) -> list[float]:
    import torch

    scores = []
    for row, (start, end) in enumerate(zip(starts, ends)):
        s_logits = student_logits[row, start - 1 : end - 1].float()
        t_logits = teacher_logits[row, start - 1 : end - 1].float()
        s_top_logits, s_ids = torch.topk(s_logits, k=TOP_K, dim=-1)
        t_top_logits, t_ids = torch.topk(t_logits, k=TOP_K, dim=-1)
        s_probs = (s_top_logits - torch.logsumexp(s_logits, dim=-1, keepdim=True)).exp()
        t_probs = (t_top_logits - torch.logsumexp(t_logits, dim=-1, keepdim=True)).exp()
        eq = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2))
        s_shared = eq.any(dim=-1)
        t_shared = eq.any(dim=-2)
        interaction = torch.minimum((s_probs * s_shared).sum(dim=-1), (t_probs * t_shared).sum(dim=-1))
        scores.append(float(torch.median(interaction)))
    return scores


def score_worker(args: tuple[Any, ...]) -> tuple[str, str]:
    (
        worker_id,
        gpu_id,
        input_path,
        student_model,
        teacher_model,
        candidate_records,
        batch_size,
        random_seed,
        high_path,
        random_path,
    ) = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frame = pd.read_parquet(input_path)
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pending = []
    for record in candidate_records:
        prompt = normalize_prompt(frame.iloc[record["source_row"]].to_dict()["prompt"])
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        for candidate in record["candidates"]:
            if candidate["valid"]:
                pending.append(
                    {
                        "source_row": record["source_row"],
                        "candidate_id": candidate["candidate_id"],
                        "token_ids": candidate["token_ids"],
                        "text": candidate["text"],
                        "prompt_ids": prompt_ids,
                    }
                )

    student = teacher = None
    scored: dict[int, list[dict[str, Any]]] = defaultdict(list)
    try:
        common = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
        student = AutoModelForCausalLM.from_pretrained(student_model, **common).eval().cuda()
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model, **common).eval().cuda()
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            sequences = [item["prompt_ids"] + item["token_ids"] for item in batch]
            padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
            input_ids = padded["input_ids"].cuda()
            attention_mask = padded["attention_mask"].cuda()
            starts = [len(item["prompt_ids"]) for item in batch]
            ends = [len(sequence) for sequence in sequences]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = student(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            scores = median_interaction_scores(student_logits, teacher_logits, starts, ends)
            del student_logits, teacher_logits
            for item, score in zip(batch, scores):
                scored[item["source_row"]].append({**item, "median_interaction_mass": score})
            print(f"[score worker {worker_id}, gpu {gpu_id}] {min(start + len(batch), len(pending))}/{len(pending)} candidates", flush=True)
    finally:
        del student, teacher
        gc.collect()
        torch.cuda.empty_cache()

    with open(high_path, "w", encoding="utf-8") as high_handle, open(random_path, "w", encoding="utf-8") as random_handle:
        for source_row, candidates in sorted(scored.items()):
            high = max(candidates, key=lambda candidate: (candidate["median_interaction_mass"], -candidate["candidate_id"]))
            chooser = random.Random(random_seed + 1_000_003 * source_row)
            random_candidate = candidates[chooser.randrange(len(candidates))]
            for handle, selection in ((high_handle, high), (random_handle, random_candidate)):
                handle.write(
                    json.dumps(
                        {
                            "source_row": source_row,
                            "candidate_id": selection["candidate_id"],
                            "prefix_token_ids": selection["token_ids"],
                            "prefix_text": selection["text"],
                            "median_interaction_mass": selection["median_interaction_mass"],
                            "valid_candidate_count": len(candidates),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return high_path, random_path


def make_dataset(frame: pd.DataFrame, selections: list[dict[str, Any]], output: Path, rule: str, config: dict[str, Any]) -> None:
    by_row = {item["source_row"]: item for item in selections}
    rows = sorted(by_row)
    output_frame = frame.iloc[rows].copy().reset_index(drop=True)
    output_frame["teacher_prefix_token_ids"] = [by_row[row]["prefix_token_ids"] for row in rows]
    output_frame["teacher_prefix_text"] = [by_row[row]["prefix_text"] for row in rows]
    output_frame["teacher_prefix_token_len"] = 128
    output_frame["teacher_prefix_finish_reason"] = "length"
    output_frame["teacher_prefix_model"] = config["student_model"]
    output_frame["teacher_prefix_max_tokens"] = 128
    output_frame["teacher_prefix_temperature"] = config["temperature"]
    output_frame["teacher_prefix_top_p"] = config["top_p"]
    output_frame["teacher_prefix_enable_thinking"] = True
    output_frame["group_opd_selector_rule"] = rule
    output_frame["group_opd_selector_score"] = [by_row[row]["median_interaction_mass"] for row in rows]
    output_frame["group_opd_candidate_id"] = [by_row[row]["candidate_id"] for row in rows]
    output_frame["group_opd_valid_candidate_count"] = [by_row[row]["valid_candidate_count"] for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_parquet(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--candidates-per-prompt", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--forward-batch-size", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if args.prefix_tokens != 128:
        raise ValueError("this pilot is intentionally fixed to Prefix128")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(args.input)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    samples = [{"source_row": index, "prompt": normalize_prompt(row["prompt"])} for index, (_, row) in enumerate(frame.iterrows())]
    shards = [[] for _ in gpu_ids]
    for index, sample in enumerate(samples):
        shards[index % len(shards)].append(sample)
    context = multiprocessing.get_context("spawn")
    gen_args = [
        (index, gpu, args.student_model, shard, args.candidates_per_prompt, args.prefix_tokens, args.temperature, args.top_p, args.generation_batch_size, str(output_dir / f"candidates_worker_{index:02d}.jsonl"))
        for index, (gpu, shard) in enumerate(zip(gpu_ids, shards))
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(gen_args), mp_context=context) as executor:
        candidate_paths = list(executor.map(generate_worker, gen_args))
    candidate_records = [record for path in candidate_paths for record in read_jsonl(Path(path))]
    score_shards = [[] for _ in gpu_ids]
    for index, record in enumerate(candidate_records):
        score_shards[index % len(score_shards)].append(record)
    score_args = [
        (index, gpu, args.input, args.student_model, args.teacher_model, shard, args.forward_batch_size, args.random_seed, str(output_dir / f"high_worker_{index:02d}.jsonl"), str(output_dir / f"random_worker_{index:02d}.jsonl"))
        for index, (gpu, shard) in enumerate(zip(gpu_ids, score_shards))
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(score_args), mp_context=context) as executor:
        selected_paths = list(executor.map(score_worker, score_args))
    high = [record for high_path, _ in selected_paths for record in read_jsonl(Path(high_path))]
    random_selected = [record for _, random_path in selected_paths for record in read_jsonl(Path(random_path))]
    if {row["source_row"] for row in high} != {row["source_row"] for row in random_selected}:
        raise RuntimeError("High and Random selections do not contain the same prompt set")
    config = vars(args) | {"input_rows": len(frame), "selected_rows": len(high)}
    make_dataset(frame, high, output_dir / "group_opd_high_median.parquet", "highest_median_interaction_mass", config)
    make_dataset(frame, random_selected, output_dir / "group_opd_random_state.parquet", "fixed_seed_random_valid_candidate", config)
    summary = {
        **config,
        "dropped_no_valid_candidate": len(frame) - len(high),
        "high_score_mean": sum(item["median_interaction_mass"] for item in high) / len(high),
        "random_score_mean": sum(item["median_interaction_mass"] for item in random_selected) / len(random_selected),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
