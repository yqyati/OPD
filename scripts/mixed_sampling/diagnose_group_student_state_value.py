#!/usr/bin/env python3
"""Offline Group OPD phase-1 diagnosis: value student-generated prefix states.

For each original prompt, sample N short student reasoning prefixes. Each valid
prefix is then completed K times by the same student and scored by the existing
rule-based verifier. The result is a within-prompt ranking of real student
continuation values, with no training update and no teacher-prefix context.
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_row(row: dict[str, Any], source_row: int) -> dict[str, Any]:
    prompt = row["prompt"]
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    answer = str(row["reward_model"]["ground_truth"])
    if not prompt:
        raise ValueError(f"row {source_row} has an empty prompt")
    return {
        "source_row": source_row,
        "dataset_index": str(row.get("extra_info", {}).get("index", source_row)),
        "prompt": prompt,
        "answer": answer,
    }


def worker_process(args: tuple[Any, ...]) -> str:
    (
        worker_id,
        gpu_id,
        student_model,
        samples,
        prefixes_per_prompt,
        prefix_tokens,
        continuations_per_prefix,
        max_tokens,
        temperature,
        top_p,
        request_batch_size,
        output_path,
    ) = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    llm = None
    try:
        llm = LLM(
            model=student_model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=2048 + prefix_tokens + max_tokens,
        )
        tokenizer = llm.get_tokenizer()
        prompt_ids = [
            tokenizer.apply_chat_template(
                sample["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for sample in samples
        ]
        prefix_params = SamplingParams(
            n=prefixes_per_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=prefix_tokens,
        )
        prefix_outputs = []
        for start in range(0, len(samples), request_batch_size):
            prefix_outputs.extend(
                llm.generate(
                    [{"prompt_token_ids": ids} for ids in prompt_ids[start : start + request_batch_size]],
                    prefix_params,
                    use_tqdm=False,
                )
            )

        state_requests = []
        invalid_states = []
        eos_id = tokenizer.eos_token_id
        for sample, ids, output in zip(samples, prompt_ids, prefix_outputs):
            for prefix_id, candidate in enumerate(output.outputs):
                prefix_ids = [int(token_id) for token_id in candidate.token_ids]
                ended = candidate.finish_reason == "stop" or (eos_id is not None and eos_id in prefix_ids)
                state = {
                    "source_row": sample["source_row"],
                    "dataset_index": sample["dataset_index"],
                    "prefix_id": prefix_id,
                    "prefix_token_ids": prefix_ids,
                    "prefix_len": len(prefix_ids),
                    "prefix_finish_reason": candidate.finish_reason,
                    "answer": sample["answer"],
                }
                if ended or len(prefix_ids) != prefix_tokens:
                    invalid_states.append({**state, "invalid_reason": "prefix_completed_before_target_len"})
                    continue
                state_requests.append({**state, "prompt_token_ids": ids + prefix_ids})

        continuation_params = SamplingParams(
            n=continuations_per_prefix,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            for state in invalid_states:
                handle.write(json.dumps({"record_type": "invalid_prefix", **state}, ensure_ascii=False) + "\n")
            for start in range(0, len(state_requests), request_batch_size):
                batch = state_requests[start : start + request_batch_size]
                outputs = llm.generate(
                    [{"prompt_token_ids": state["prompt_token_ids"]} for state in batch],
                    continuation_params,
                    use_tqdm=False,
                )
                for state, output in zip(batch, outputs):
                    for continuation_id, candidate in enumerate(output.outputs):
                        response = candidate.text
                        handle.write(
                            json.dumps(
                                {
                                    "record_type": "continuation",
                                    "source_row": state["source_row"],
                                    "dataset_index": state["dataset_index"],
                                    "prefix_id": state["prefix_id"],
                                    "prefix_len": state["prefix_len"],
                                    "prefix_token_ids": state["prefix_token_ids"],
                                    "prefix_finish_reason": state["prefix_finish_reason"],
                                    "continuation_id": continuation_id,
                                    "answer": state["answer"],
                                    "response": response,
                                    "response_token_ids": [int(token_id) for token_id in candidate.token_ids],
                                    "response_length": len(candidate.token_ids),
                                    "rule_score": int(grade_answer_verl(response, state["answer"])),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                print(
                    f"[worker {worker_id}, gpu {gpu_id}] completed {min(start + len(batch), len(state_requests))}/{len(state_requests)} valid states",
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--prefixes-per-prompt", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--continuations-per-prefix", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--request-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.num_prompts, args.prefixes_per_prompt, args.prefix_tokens, args.continuations_per_prefix) <= 0:
        raise ValueError("all count arguments must be positive")
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(args.input)
    rows = list(range(len(frame)))
    random.Random(args.seed).shuffle(rows)
    selected_rows = rows[: min(args.num_prompts, len(rows))]
    samples = [compact_row(frame.iloc[row].to_dict(), row) for row in selected_rows]
    config = vars(args) | {"selected_source_rows": selected_rows}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    shards = [[] for _ in gpu_ids]
    for index, sample in enumerate(samples):
        shards[index % len(gpu_ids)].append(sample)
    worker_args = [
        (
            worker_id,
            gpu_id,
            args.student_model,
            shard,
            args.prefixes_per_prompt,
            args.prefix_tokens,
            args.continuations_per_prefix,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.request_batch_size,
            str(output_dir / f"group_state_worker_{worker_id:02d}.jsonl"),
        )
        for worker_id, (gpu_id, shard) in enumerate(zip(gpu_ids, shards))
        if shard
    ]
    print(
        f"group state diagnosis: prompts={len(samples)}, n={args.prefixes_per_prompt}, "
        f"K={args.continuations_per_prefix}, GPUs={gpu_ids}",
        flush=True,
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        paths = list(executor.map(worker_process, worker_args))

    raw_records = [record for path in paths for record in load_jsonl(Path(path))]
    continuation_records = [record for record in raw_records if record["record_type"] == "continuation"]
    invalid_records = [record for record in raw_records if record["record_type"] == "invalid_prefix"]
    by_state: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in continuation_records:
        by_state[(record["source_row"], record["prefix_id"])].append(record)

    state_values = []
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (source_row, prefix_id), records in sorted(by_state.items()):
        value = sum(record["rule_score"] for record in records) / len(records)
        state = {
            "source_row": source_row,
            "dataset_index": records[0]["dataset_index"],
            "prefix_id": prefix_id,
            "prefix_len": records[0]["prefix_len"],
            "V_student": value,
            "num_continuations": len(records),
            "mean_response_length": sum(record["response_length"] for record in records) / len(records),
        }
        state_values.append(state)
        by_prompt[source_row].append(state)

    rank_summary = []
    for source_row, states in sorted(by_prompt.items()):
        best_value = max(state["V_student"] for state in states)
        worst_value = min(state["V_student"] for state in states)
        rank_summary.append(
            {
                "source_row": source_row,
                "dataset_index": states[0]["dataset_index"],
                "valid_prefixes": len(states),
                "best_prefix_ids": [state["prefix_id"] for state in states if state["V_student"] == best_value],
                "worst_prefix_ids": [state["prefix_id"] for state in states if state["V_student"] == worst_value],
                "best_V_student": best_value,
                "worst_V_student": worst_value,
                "value_gap": best_value - worst_value,
            }
        )

    summary = {
        "num_prompts": len(samples),
        "valid_student_prefix_states": len(state_values),
        "invalid_early_completed_prefixes": len(invalid_records),
        "num_full_continuations": len(continuation_records),
        "mean_state_value": sum(state["V_student"] for state in state_values) / len(state_values) if state_values else None,
        "prompts_with_nonzero_within_group_value_gap": sum(item["value_gap"] > 0 for item in rank_summary),
        "mean_within_group_value_gap": sum(item["value_gap"] for item in rank_summary) / len(rank_summary) if rank_summary else None,
    }
    with open(output_dir / "group_prefix_continuations.jsonl", "w", encoding="utf-8") as handle:
        for record in continuation_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / "group_prefix_state_values.jsonl", "w", encoding="utf-8") as handle:
        for record in state_values:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / "group_prefix_rankings.jsonl", "w", encoding="utf-8") as handle:
        for record in rank_summary:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
