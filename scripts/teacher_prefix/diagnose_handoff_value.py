#!/usr/bin/env python3
"""Measure student continuation value at several exact teacher-prefix handoffs.

This is an offline diagnostic only. It does not import training code or update a
model. For each selected training prompt and candidate prefix length, it feeds
the student the original chat-template token IDs plus the exact generated
teacher-prefix token IDs, completes the response, and grades the result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

EVAL_UTILS_DIR = Path(__file__).resolve().parents[1] / "val" / "eval"
sys.path.insert(0, str(EVAL_UTILS_DIR))
from utils import grade_answer_verl  # noqa: E402


def parse_lengths(value: str) -> list[int]:
    lengths = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not lengths or lengths[0] < 0:
        raise argparse.ArgumentTypeError("--prefix-lengths must contain non-negative integers")
    return lengths


def compact_row(row: dict[str, Any], source_row: int) -> dict[str, Any]:
    reward_model = row["reward_model"]
    prompt = row["prompt"]
    prefix_ids = [int(token_id) for token_id in row["teacher_prefix_token_ids"]]
    if not prompt or not prefix_ids:
        raise ValueError(f"row {source_row} has an empty prompt or teacher prefix")
    return {
        "source_row": source_row,
        "prompt": prompt,
        "answer": str(reward_model["ground_truth"]),
        "dataset_index": str(row.get("extra_info", {}).get("index", source_row)),
        "teacher_prefix_token_ids": prefix_ids,
    }


def build_requests(tokenizer: Any, samples: list[dict[str, Any]], lengths: list[int]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for sample in samples:
        prompt_ids = tokenizer.apply_chat_template(
            sample["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prefix_ids = sample["teacher_prefix_token_ids"]
        for requested_len in lengths:
            effective_len = min(requested_len, len(prefix_ids))
            requests.append(
                {
                    "prompt_token_ids": prompt_ids + prefix_ids[:effective_len],
                    "sample": sample,
                    "requested_prefix_len": requested_len,
                    "effective_prefix_len": effective_len,
                }
            )
    return requests


def worker_process(args: tuple[Any, ...]) -> str:
    (
        worker_id,
        gpu_id,
        model_path,
        samples,
        lengths,
        num_continuations,
        max_tokens,
        temperature,
        top_p,
        request_batch_size,
        max_model_len,
        output_path,
    ) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    llm = None
    try:
        print(f"[worker {worker_id}, gpu {gpu_id}] loading student model", flush=True)
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=max_model_len,
        )
        tokenizer = llm.get_tokenizer()
        requests = build_requests(tokenizer, samples, lengths)
        sampling = SamplingParams(
            n=num_continuations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        with open(output_path, "w", encoding="utf-8") as handle:
            for start in range(0, len(requests), request_batch_size):
                batch = requests[start : start + request_batch_size]
                prompts = [{"prompt_token_ids": request["prompt_token_ids"]} for request in batch]
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                for request, output in zip(batch, outputs):
                    sample = request["sample"]
                    for continuation_id, candidate in enumerate(output.outputs):
                        response = candidate.text
                        record = {
                            "source_row": sample["source_row"],
                            "dataset_index": sample["dataset_index"],
                            "requested_prefix_len": request["requested_prefix_len"],
                            "effective_prefix_len": request["effective_prefix_len"],
                            "prefix_available_len": len(sample["teacher_prefix_token_ids"]),
                            "continuation_id": continuation_id,
                            "answer": sample["answer"],
                            "response": response,
                            "response_token_ids": [int(token_id) for token_id in candidate.token_ids],
                            "response_length": len(candidate.token_ids),
                            "rule_score": int(grade_answer_verl(response, sample["answer"])),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"[worker {worker_id}, gpu {gpu_id}] completed {min(start + len(batch), len(requests))}/{len(requests)} candidate states",
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


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def summarize(records: list[dict[str, Any]], output_dir: Path, requested_lengths: list[int]) -> None:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_length[record["requested_prefix_len"]].append(record)
        by_prompt[record["source_row"]][record["requested_prefix_len"]].append(record)

    length_summary = []
    for length in requested_lengths:
        group = by_length[length]
        if not group:
            continue
        length_summary.append(
            {
                "requested_prefix_len": length,
                "num_rollouts": len(group),
                "num_prompts": len({record["source_row"] for record in group}),
                "mean_rule_reward": sum(record["rule_score"] for record in group) / len(group),
                "mean_response_length": sum(record["response_length"] for record in group) / len(group),
            }
        )

    prompt_summary = []
    winner_counts: dict[int, float] = defaultdict(float)
    for source_row, groups in sorted(by_prompt.items()):
        values = {
            length: sum(record["rule_score"] for record in group) / len(group)
            for length, group in groups.items()
        }
        best_value = max(values.values())
        winners = [length for length, value in values.items() if value == best_value]
        for winner in winners:
            winner_counts[winner] += 1.0 / len(winners)
        first_record = next(iter(next(iter(groups.values()))))
        prompt_summary.append(
            {
                "source_row": source_row,
                "dataset_index": first_record["dataset_index"],
                "continuation_value_by_prefix_len": values,
                "best_prefix_lengths": winners,
                "best_continuation_value": best_value,
            }
        )

    with open(output_dir / "handoff_rollouts.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / "per_prompt_handoff_values.jsonl", "w", encoding="utf-8") as handle:
        for record in prompt_summary:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "num_rollouts": len(records),
        "num_prompts": len(by_prompt),
        "requested_prefix_lengths": requested_lengths,
        "by_prefix_length": length_summary,
        "best_length_tie_split_counts": {str(length): winner_counts[length] for length in requested_lengths},
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Teacher-prefix parquet with exact teacher_prefix_token_ids.")
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix-lengths", type=parse_lengths, default=parse_lengths("0,128,256,512,1024"))
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--continuations-per-prefix", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--request-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selected-source-rows-from",
        default=None,
        help="Optional config.json whose selected_source_rows are reused exactly instead of sampling a new subset.",
    )
    parser.add_argument(
        "--require-alive-at-prefix-len",
        type=int,
        default=None,
        help=(
            "Restrict sampled rows to teacher trajectories that remain live at this exact length. "
            "A trajectory ending with EOS at or before this position is excluded."
        ),
    )
    args = parser.parse_args()

    if args.num_prompts <= 0 or args.continuations_per_prefix <= 0:
        raise ValueError("--num-prompts and --continuations-per-prefix must be positive")
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(args.input)
    if "teacher_prefix_token_ids" not in frame.columns:
        raise ValueError("input has no teacher_prefix_token_ids column")
    eligible_rows = list(range(len(frame)))
    if args.require_alive_at_prefix_len is not None:
        required_len = args.require_alive_at_prefix_len
        if required_len <= 0:
            raise ValueError("--require-alive-at-prefix-len must be positive")
        if max(args.prefix_lengths) > required_len:
            raise ValueError("all requested prefix lengths must be <= --require-alive-at-prefix-len")
        if "teacher_prefix_token_len" not in frame.columns or "teacher_prefix_finish_reason" not in frame.columns:
            raise ValueError("alive-only filtering requires teacher_prefix_token_len and teacher_prefix_finish_reason")
        eligible_rows = [
            row_index
            for row_index, row in frame.iterrows()
            if int(row["teacher_prefix_token_len"]) >= required_len
            and not (
                str(row["teacher_prefix_finish_reason"]) == "stop"
                and int(row["teacher_prefix_token_len"]) <= required_len
            )
        ]
        if not eligible_rows:
            raise RuntimeError(f"no teacher trajectories remain live at prefix length {required_len}")
    if args.selected_source_rows_from is not None:
        selected_config = json.loads(Path(args.selected_source_rows_from).read_text())
        selected_rows = [int(row) for row in selected_config["selected_source_rows"]]
        eligible_set = set(eligible_rows)
        invalid_rows = [row for row in selected_rows if row not in eligible_set]
        if invalid_rows:
            raise RuntimeError(f"selected-source config contains {len(invalid_rows)} rows outside current eligibility filter")
        if args.num_prompts != len(selected_rows):
            raise ValueError(
                f"--num-prompts={args.num_prompts} does not match selected-source config size {len(selected_rows)}"
            )
    else:
        rng = random.Random(args.seed)
        source_rows = eligible_rows
        rng.shuffle(source_rows)
        selected_rows = source_rows[: min(args.num_prompts, len(source_rows))]
    samples = [compact_row(frame.iloc[row].to_dict(), row) for row in selected_rows]

    config = vars(args) | {
        "eligible_source_rows": len(eligible_rows),
        "selected_source_rows": selected_rows,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    sample_shards = [[] for _ in gpu_ids]
    for index, sample in enumerate(samples):
        sample_shards[index % len(gpu_ids)].append(sample)
    max_model_len = 2048 + max(args.prefix_lengths) + args.max_tokens
    worker_args = []
    for worker_id, (gpu_id, shard) in enumerate(zip(gpu_ids, sample_shards)):
        if not shard:
            continue
        worker_args.append(
            (
                worker_id,
                gpu_id,
                args.student_model,
                shard,
                args.prefix_lengths,
                args.continuations_per_prefix,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.request_batch_size,
                max_model_len,
                str(output_dir / f"handoff_rollouts_worker_{worker_id:02d}.jsonl"),
            )
        )

    print(
        f"diagnosing {len(samples)} prompts x {len(args.prefix_lengths)} lengths x "
        f"{args.continuations_per_prefix} continuations on GPUs {gpu_ids}",
        flush=True,
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        worker_outputs = list(executor.map(worker_process, worker_args))
    summarize(load_records([Path(path) for path in worker_outputs]), output_dir, args.prefix_lengths)


if __name__ == "__main__":
    main()
