#!/usr/bin/env python3
"""Test gold-answer recoverability as a no-rollout teacher-prefix heuristic.

For each exact teacher-prefix state, append the same hidden final-answer probe
containing the training example's gold answer. The score is the current
student's average conditional log probability over that probe. The gold answer
is never passed to rollout or training context; it is used only as an offline
diagnostic scalar, analogous to a verifier reward.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import math
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in range(start, end):
            ranks[order[index]] = rank
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def canonical_answer(answer: str) -> str:
    answer = answer.strip()
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        return answer[len("\\boxed{") : -1]
    return answer


def extract_actual_logprobs(output: Any, prompt_length: int, scored_token_ids: list[int]) -> list[float]:
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None:
        raise RuntimeError("vLLM returned no prompt_logprobs")
    start = prompt_length - len(scored_token_ids)
    scores = []
    for offset, token_id in enumerate(scored_token_ids):
        options = prompt_logprobs[start + offset]
        if not options or token_id not in options:
            raise RuntimeError(f"missing actual answer-probe token logprob at offset={offset}")
        scores.append(float(options[token_id].logprob))
    return scores


def worker_process(args: tuple[Any, ...]) -> str:
    worker_id, gpu_id, student_model, samples, prefix_lengths, batch_size, output_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    llm = None
    try:
        llm = LLM(
            model=student_model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )
        tokenizer = llm.get_tokenizer()
        requests = []
        for sample in samples:
            prompt_ids = tokenizer.apply_chat_template(
                sample["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            answer_text = canonical_answer(sample["answer"])
            probe_ids = tokenizer.encode(
                "\n\nTherefore, the final answer is \\boxed{" + answer_text + "}",
                add_special_tokens=False,
            )
            for requested_length in prefix_lengths:
                effective_length = min(requested_length, len(sample["prefix_ids"]))
                full_ids = prompt_ids + sample["prefix_ids"][:effective_length] + probe_ids
                requests.append(
                    {
                        "source_row": sample["source_row"],
                        "requested_prefix_len": requested_length,
                        "effective_prefix_len": effective_length,
                        "prompt_token_ids": full_ids,
                        "probe_token_ids": probe_ids,
                    }
                )

        scoring_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        with open(output_path, "w", encoding="utf-8") as handle:
            for start in range(0, len(requests), batch_size):
                batch = requests[start : start + batch_size]
                outputs = llm.generate(
                    [{"prompt_token_ids": request["prompt_token_ids"]} for request in batch],
                    scoring_params,
                    use_tqdm=False,
                )
                for request, output in zip(batch, outputs):
                    token_scores = extract_actual_logprobs(
                        output,
                        len(request["prompt_token_ids"]),
                        request["probe_token_ids"],
                    )
                    record = {
                        "source_row": request["source_row"],
                        "requested_prefix_len": request["requested_prefix_len"],
                        "effective_prefix_len": request["effective_prefix_len"],
                        "answer_probe_tokens": len(token_scores),
                        "gold_answer_recoverability": mean(token_scores),
                    }
                    handle.write(json.dumps(record) + "\n")
                print(
                    f"[worker {worker_id}, gpu {gpu_id}] completed {min(start + len(batch), len(requests))}/{len(requests)} states",
                    flush=True,
                )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--handoff-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    handoff_dir = Path(args.handoff_dir)
    config = json.loads((handoff_dir / "config.json").read_text())
    handoff_records = load_jsonl(handoff_dir / "per_prompt_handoff_values.jsonl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prefix_lengths = [int(length) for length in config["prefix_lengths"]]
    outcomes = {
        (int(record["source_row"]), int(length)): value
        for record in handoff_records
        for length, value in record["continuation_value_by_prefix_len"].items()
    }
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU")

    frame = pd.read_parquet(args.input)
    samples = []
    for source_row in config["selected_source_rows"]:
        row = frame.iloc[int(source_row)].to_dict()
        samples.append(
            {
                "source_row": int(source_row),
                "prompt": row["prompt"],
                "answer": str(row["reward_model"]["ground_truth"]),
                "prefix_ids": [int(token_id) for token_id in row["teacher_prefix_token_ids"]],
            }
        )

    shards = [[] for _ in gpu_ids]
    for index, sample in enumerate(samples):
        shards[index % len(gpu_ids)].append(sample)
    worker_args = [
        (worker_id, gpu_id, args.student_model, shard, prefix_lengths, args.batch_size, str(output_dir / f"recoverability_worker_{worker_id:02d}.jsonl"))
        for worker_id, (gpu_id, shard) in enumerate(zip(gpu_ids, shards))
        if shard
    ]
    print(f"scoring {len(samples)} prompts x {len(prefix_lengths)} prefix states", flush=True)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        paths = list(executor.map(worker_process, worker_args))
    records = [record for path in paths for record in load_jsonl(Path(path))]

    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        record["V_student"] = outcomes[(record["source_row"], record["requested_prefix_len"])]
        by_prompt[record["source_row"]].append(record)
        by_length[record["requested_prefix_len"]].append(record)

    values = [record["gold_answer_recoverability"] for record in records]
    outcome_values = [record["V_student"] for record in records]
    all_tie_prompts = 0
    nontrivial_prompts = 0
    nontrivial_hits = 0
    all_hits = 0
    heuristic_winner_counts: dict[int, int] = defaultdict(int)
    for rows in by_prompt.values():
        best_score = max(row["gold_answer_recoverability"] for row in rows)
        heuristic_winners = {row["requested_prefix_len"] for row in rows if row["gold_answer_recoverability"] == best_score}
        for winner in heuristic_winners:
            heuristic_winner_counts[winner] += 1
        best_outcome = max(row["V_student"] for row in rows)
        outcome_winners = {row["requested_prefix_len"] for row in rows if row["V_student"] == best_outcome}
        hit = bool(heuristic_winners.intersection(outcome_winners))
        all_hits += hit
        if len(outcome_winners) == len(rows):
            all_tie_prompts += 1
        else:
            nontrivial_prompts += 1
            nontrivial_hits += hit

    summary = {
        "num_prompts": len(by_prompt),
        "by_prefix_length": [
            {
                "prefix_length": length,
                "mean_gold_answer_recoverability": mean(row["gold_answer_recoverability"] for row in by_length[length]),
                "mean_V_student": mean(row["V_student"] for row in by_length[length]),
            }
            for length in prefix_lengths
        ],
        "global_spearman_recoverability_vs_V_student": spearman(values, outcome_values),
        "global_pearson_recoverability_vs_V_student": pearson(values, outcome_values),
        "all_prompt_top1_hit_rate_including_outcome_ties": all_hits / len(by_prompt),
        "all_zero_tie_prompts": all_tie_prompts,
        "nontrivial_prompt_top1_hit_rate": nontrivial_hits / nontrivial_prompts if nontrivial_prompts else None,
        "nontrivial_prompts": nontrivial_prompts,
        "heuristic_argmax_count": {str(length): heuristic_winner_counts[length] for length in prefix_lengths},
    }
    with open(output_dir / "gold_answer_recoverability.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
