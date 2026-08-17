#!/usr/bin/env python3
"""Batched, resumable Qwen3 generation for BigCodeBench-Instruct.

This intentionally does not use BigCodeBench's stock vLLM provider: that
provider calls ``apply_chat_template`` without Qwen3's ``enable_thinking``
argument.  The data, sanitizer, output schema, and evaluator remain the
official BigCodeBench ones.

Run one process per GPU (tensor_parallel_size=1), then merge the shards.
"""

import argparse
import json
import re
from pathlib import Path

from bigcodebench.data import get_bigcodebench
from bigcodebench.sanitize import sanitize
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
# Protect the official sanitizer from pathological prose-only completions.  A
# normal BigCodeBench answer is passed to the official sanitizer unchanged.
MAX_SANITIZE_CHARS = 24_000
MAX_SANITIZE_LINES = 1_200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=7168)
    parser.add_argument("--request-chunk-size", type=int, default=8)
    parser.add_argument("--subset", choices=("full", "hard"), default="full")
    parser.add_argument("--id-range", help="Half-open global task-index range, e.g. 0-8")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Qwen3 native enable_thinking=True (required for Base).",
    )
    return parser.parse_args()


def parse_id_range(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    parts = raw.split("-", 1)
    if len(parts) != 2:
        raise ValueError("--id-range must be START-END")
    low, high = (int(x) for x in parts)
    if low < 0 or high <= low:
        raise ValueError("--id-range must satisfy 0 <= START < END")
    return low, high


def build_prompt(task: dict, tokenizer, enable_thinking: bool) -> str:
    """Official Instruct task text inside the native Qwen3 chat template."""
    if tokenizer.chat_template is None:
        raise ValueError("BigCodeBench-Instruct requires a model chat template")
    content = f"{INSTRUCTION_PREFIX}\n{task['instruct_prompt'].strip()}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def safe_sanitize(text: str, entry_point: str) -> str:
    if len(text) <= MAX_SANITIZE_CHARS and text.count("\n") <= MAX_SANITIZE_LINES:
        return sanitize(text, entry_point)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (blocks[-1] if blocks else text)[:MAX_SANITIZE_CHARS]


def shard_path(output: Path, rank: int) -> Path:
    return output.parent / f"{output.stem}.rank{rank}{output.suffix}"


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            keys.add((row["task_id"], int(row["sample_id"])))
    return keys


def generate(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("--model is required unless --merge is used")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world-size)")
    if args.n_samples < 1 or args.request_chunk_size < 1:
        raise ValueError("--n-samples and --request-chunk-size must be positive")

    output = shard_path(args.output, args.rank)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output)
    selected = parse_id_range(args.id_range)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    tasks = list(get_bigcodebench(subset=args.subset).items())

    requests = []
    for index, (task_id, task) in enumerate(tasks):
        if selected is not None and not selected[0] <= index < selected[1]:
            continue
        if index % args.world_size != args.rank:
            continue
        prompt = build_prompt(task, tokenizer, args.enable_thinking)
        for sample_id in range(args.n_samples):
            if (task_id, sample_id) not in done:
                requests.append((task_id, sample_id, task["entry_point"], prompt))

    print(f"rank={args.rank}: {len(requests)} pending BigCodeBench requests", flush=True)
    if not requests:
        return
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    with output.open("a") as handle:
        for start in range(0, len(requests), args.request_chunk_size):
            batch = requests[start : start + args.request_chunk_size]
            results = llm.generate([item[3] for item in batch], sampling, use_tqdm=True)
            for (task_id, sample_id, entry_point, _), result in zip(batch, results, strict=True):
                raw = result.outputs[0].text
                handle.write(json.dumps({
                    "task_id": task_id,
                    "sample_id": sample_id,
                    "solution": safe_sanitize(raw, entry_point),
                    "raw_solution": raw,
                }) + "\n")
            handle.flush()
            print(f"rank={args.rank}: wrote {start + len(batch)}/{len(requests)}", flush=True)


def merge(args: argparse.Namespace) -> None:
    tasks = list(get_bigcodebench(subset=args.subset))
    selected = parse_id_range(args.id_range)
    selected_ids = {
        task_id for index, task_id in enumerate(tasks)
        if selected is None or selected[0] <= index < selected[1]
    }
    rows = {}
    for rank in range(args.world_size):
        path = shard_path(args.output, rank)
        if not path.exists():
            raise FileNotFoundError(f"missing shard: {path}")
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(row["task_id"], int(row["sample_id"]))] = row
    expected = len(selected_ids) * args.n_samples
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} samples, found {len(rows)}")
    with args.output.open("w") as handle:
        for _, row in sorted(rows.items()):
            handle.write(json.dumps(row) + "\n")
    print(f"merged {len(rows)} samples into {args.output}", flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    merge(arguments) if arguments.merge else generate(arguments)
