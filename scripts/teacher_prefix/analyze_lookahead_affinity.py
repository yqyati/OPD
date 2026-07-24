#!/usr/bin/env python3
"""Test a no-rollout prefix heuristic against measured student handoff value.

For a candidate handoff length l, Lookahead Teacher-Trajectory Affinity is the
mean joint log probability that teacher and student assign to the next W exact
teacher-generated tokens. It uses one teacher extension only to score l=1024;
all shorter candidates reuse the already cached teacher Prefix1024 trace.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import math
import multiprocessing
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_norm == 0 or y_norm == 0:
        return None
    return numerator / (x_norm * y_norm)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def compact_row(row: dict[str, Any], source_row: int) -> dict[str, Any]:
    return {
        "source_row": source_row,
        "prompt": row["prompt"],
        "prefix_ids": [int(token_id) for token_id in row["teacher_prefix_token_ids"]],
        "temperature": float(row.get("teacher_prefix_temperature", 0.7)),
        "top_p": float(row.get("teacher_prefix_top_p", 0.95)),
    }


def extract_prompt_logps(output: Any, prompt_length: int, trace_ids: list[int]) -> list[float]:
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None:
        raise RuntimeError("vLLM returned no prompt_logprobs")
    scores: list[float] = []
    for position, token_id in enumerate(trace_ids):
        absolute_position = prompt_length + position
        if absolute_position >= len(prompt_logprobs):
            raise RuntimeError(
                f"prompt_logprobs is too short: position={absolute_position}, length={len(prompt_logprobs)}"
            )
        options = prompt_logprobs[absolute_position]
        if not options or token_id not in options:
            raise RuntimeError(f"missing actual prompt token logprob at trace position {position}")
        scores.append(float(options[token_id].logprob))
    return scores


def score_prompts(llm: Any, prompts: list[list[int]], prompt_lengths: list[int], traces: list[list[int]]) -> list[list[float]]:
    from vllm import SamplingParams

    scoring_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        scoring_params,
        use_tqdm=False,
    )
    return [
        extract_prompt_logps(output, prompt_length, trace)
        for output, prompt_length, trace in zip(outputs, prompt_lengths, traces)
    ]


def worker_process(args: tuple[Any, ...]) -> str:
    (
        worker_id,
        gpu_id,
        teacher_model,
        student_model,
        samples,
        prefix_lengths,
        lookahead_tokens,
        batch_size,
        output_path,
    ) = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    teacher = None
    student = None
    try:
        print(f"[worker {worker_id}, gpu {gpu_id}] loading teacher", flush=True)
        max_model_len = 2048 + 1024 + lookahead_tokens + 1
        teacher = LLM(
            model=teacher_model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=max_model_len,
        )
        teacher_tokenizer = teacher.get_tokenizer()

        teacher_prompt_ids = []
        prefix_contexts = []
        for sample in samples:
            prompt_ids = teacher_tokenizer.apply_chat_template(
                sample["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            teacher_prompt_ids.append(prompt_ids)
            prefix_contexts.append(prompt_ids + sample["prefix_ids"])

        extensions: list[list[int]] = []
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            params = SamplingParams(
                temperature=batch_samples[0]["temperature"],
                top_p=batch_samples[0]["top_p"],
                max_tokens=lookahead_tokens,
                ignore_eos=True,
            )
            outputs = teacher.generate(
                [{"prompt_token_ids": prompt} for prompt in prefix_contexts[start : start + batch_size]],
                params,
                use_tqdm=False,
            )
            extensions.extend([[int(token_id) for token_id in output.outputs[0].token_ids] for output in outputs])

        traces = [sample["prefix_ids"] + extension for sample, extension in zip(samples, extensions)]
        teacher_full_prompts = [prompt + trace for prompt, trace in zip(teacher_prompt_ids, traces)]
        teacher_logps: list[list[float]] = []
        for start in range(0, len(samples), batch_size):
            teacher_logps.extend(
                score_prompts(
                    teacher,
                    teacher_full_prompts[start : start + batch_size],
                    [len(ids) for ids in teacher_prompt_ids[start : start + batch_size]],
                    traces[start : start + batch_size],
                )
            )

        del teacher
        teacher = None
        gc.collect()
        import torch

        torch.cuda.empty_cache()

        print(f"[worker {worker_id}, gpu {gpu_id}] loading student", flush=True)
        student = LLM(
            model=student_model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=max_model_len,
        )
        student_tokenizer = student.get_tokenizer()
        student_prompt_ids = [
            student_tokenizer.apply_chat_template(
                sample["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for sample in samples
        ]
        student_full_prompts = [prompt + trace for prompt, trace in zip(student_prompt_ids, traces)]
        student_logps: list[list[float]] = []
        for start in range(0, len(samples), batch_size):
            student_logps.extend(
                score_prompts(
                    student,
                    student_full_prompts[start : start + batch_size],
                    [len(ids) for ids in student_prompt_ids[start : start + batch_size]],
                    traces[start : start + batch_size],
                )
            )

        with open(output_path, "w", encoding="utf-8") as handle:
            for sample, trace, teacher_scores, student_scores in zip(samples, traces, teacher_logps, student_logps):
                for length in prefix_lengths:
                    # Prefix generation may stop before its requested maximum.
                    # Match the handoff diagnosis by scoring from the effective
                    # prefix state, then use the forced-length extension when
                    # the requested handoff reaches the generated prefix end.
                    effective_length = min(length, len(sample["prefix_ids"]))
                    end = effective_length + lookahead_tokens
                    if end > len(trace):
                        raise RuntimeError(
                            f"trace too short for requested_length={length}, "
                            f"effective_length={effective_length}, window={lookahead_tokens}"
                        )
                    teacher_window = teacher_scores[effective_length:end]
                    student_window = student_scores[effective_length:end]
                    record = {
                        "source_row": sample["source_row"],
                        "requested_prefix_len": length,
                        "effective_prefix_len": effective_length,
                        "lookahead_tokens": lookahead_tokens,
                        "teacher_mean_logprob": mean(teacher_window),
                        "student_mean_logprob": mean(student_window),
                        "joint_affinity": (mean(teacher_window) + mean(student_window)) / 2.0,
                    }
                    handle.write(json.dumps(record) + "\n")
        print(f"[worker {worker_id}, gpu {gpu_id}] completed {len(samples)} prompts", flush=True)
    finally:
        if teacher is not None:
            del teacher
        if student is not None:
            del student
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--handoff-dir", required=True, help="Completed diagnose_handoff_value.py output directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lookahead-tokens", type=int, default=128)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    handoff_dir = Path(args.handoff_dir)
    config = json.loads((handoff_dir / "config.json").read_text())
    handoff_records = load_jsonl(handoff_dir / "per_prompt_handoff_values.jsonl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prefix_lengths = [int(length) for length in config["prefix_lengths"]]
    if max(prefix_lengths) != 1024:
        raise ValueError("this analysis currently expects Prefix1024 source data")
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU")

    outcomes = {
        (int(record["source_row"]), int(length)): value
        for record in handoff_records
        for length, value in record["continuation_value_by_prefix_len"].items()
    }
    frame = pd.read_parquet(args.input)
    samples = [compact_row(frame.iloc[int(row)].to_dict(), int(row)) for row in config["selected_source_rows"]]
    sample_shards = [[] for _ in gpu_ids]
    for index, sample in enumerate(samples):
        sample_shards[index % len(gpu_ids)].append(sample)
    worker_args = [
        (
            worker_id,
            gpu_id,
            args.teacher_model,
            args.student_model,
            shard,
            prefix_lengths,
            args.lookahead_tokens,
            args.batch_size,
            str(output_dir / f"affinity_worker_{worker_id:02d}.jsonl"),
        )
        for worker_id, (gpu_id, shard) in enumerate(zip(gpu_ids, sample_shards))
        if shard
    ]
    print(f"scoring {len(samples)} prompts on GPUs {gpu_ids}", flush=True)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        output_paths = list(executor.map(worker_process, worker_args))
    records = [record for path in output_paths for record in load_jsonl(Path(path))]

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        record["V_student"] = outcomes[(record["source_row"], record["requested_prefix_len"])]
        by_length[record["requested_prefix_len"]].append(record)
        by_prompt[record["source_row"]].append(record)

    by_length_summary = []
    for length in prefix_lengths:
        rows = by_length[length]
        by_length_summary.append(
            {
                "prefix_length": length,
                "mean_teacher_logprob": mean(row["teacher_mean_logprob"] for row in rows),
                "mean_student_logprob": mean(row["student_mean_logprob"] for row in rows),
                "mean_joint_affinity": mean(row["joint_affinity"] for row in rows),
                "mean_V_student": mean(row["V_student"] for row in rows),
            }
        )

    affinity_values = [record["joint_affinity"] for record in records]
    outcome_values = [record["V_student"] for record in records]
    top1_hits = 0
    for rows in by_prompt.values():
        best_affinity = max(row["joint_affinity"] for row in rows)
        affinity_winners = {row["requested_prefix_len"] for row in rows if row["joint_affinity"] == best_affinity}
        best_outcome = max(row["V_student"] for row in rows)
        outcome_winners = {row["requested_prefix_len"] for row in rows if row["V_student"] == best_outcome}
        top1_hits += bool(affinity_winners.intersection(outcome_winners))

    summary = {
        "num_prompts": len(by_prompt),
        "lookahead_tokens": args.lookahead_tokens,
        "by_prefix_length": by_length_summary,
        "global_spearman_joint_affinity_vs_V_student": spearman(affinity_values, outcome_values),
        "global_pearson_joint_affinity_vs_V_student": pearson(affinity_values, outcome_values),
        "per_prompt_top1_hit_rate_including_outcome_ties": top1_hits / len(by_prompt),
    }
    with open(output_dir / "lookahead_affinity.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
