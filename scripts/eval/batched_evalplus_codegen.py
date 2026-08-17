#!/usr/bin/env python3
"""Batched, resumable vLLM generation for EvalPlus.

Each worker owns a disjoint task shard and submits many requests at once. This
keeps vLLM's continuous batch scheduler full while preserving EvalPlus's
official prompt construction and sanitized JSONL schema.
"""

import argparse
import json
import re
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import make_raw_chat_prompt
from evalplus.sanitize import sanitize
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)
MAX_SANITIZE_CHARS = 6000
EURUS_CODE_SUFFIX = (
    "\n\nWrite Python code to solve the problem. Present the code in \n"
    "```python\nYour code\n```\nat the end."
)


def safe_sanitize(solution: str, entry_point: str) -> str:
    """Avoid EvalPlus's quadratic sanitizer for pathological base-model output."""
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", solution, flags=re.DOTALL | re.IGNORECASE)
    fallback = (code_blocks[-1] if code_blocks else solution)[:MAX_SANITIZE_CHARS]
    if len(solution) > MAX_SANITIZE_CHARS or len(solution.splitlines()) > 300:
        return fallback
    try:
        return sanitize(solution, entrypoint=entry_point)
    except Exception:
        return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # Keep the benchmark generation budget explicit and stable across resumed
    # shards. Smaller request batches prevent a few pathological completions
    # from consuming all host memory before their shard is saved.
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--request-chunk-size", type=int, default=128)
    parser.add_argument(
        "--prompt-contract",
        choices=("evalplus", "eurus"),
        default="evalplus",
        help="evalplus adds its reasoning instruction; eurus reproduces the code-training suffix exactly.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the model's native thinking mode in the chat template.",
    )
    return parser.parse_args()


def load_dataset(name: str):
    return get_human_eval_plus() if name == "humaneval" else get_mbpp_plus()


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed.add((row["task_id"], int(row["sample_id"])))
    return completed


def build_prompt(task: dict, tokenizer, enable_thinking: bool, prompt_contract: str) -> tuple[str, bool]:
    task_prompt = task["prompt"].strip()
    direct_completion = tokenizer.chat_template is None
    if direct_completion:
        return task_prompt + "\n", True
    if prompt_contract == "eurus":
        task_prompt += EURUS_CODE_SUFFIX
        return (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            ),
            False,
        )
    return (
        make_raw_chat_prompt(
            task_prompt,
            instruction_prefix=INSTRUCTION_PREFIX,
            response_prefix=RESPONSE_PREFIX,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        ),
        False,
    )


def write_records(path: Path, requests, outputs) -> None:
    with path.open("a") as handle:
        for request, output in zip(requests, outputs, strict=True):
            raw_text = output.outputs[0].text
            solution = request["prompt"] + raw_text if request["direct_completion"] else raw_text
            handle.write(
                json.dumps(
                    {
                        "task_id": request["task_id"],
                        "sample_id": request["sample_id"],
                        "solution": safe_sanitize(solution, entry_point=request["entry_point"]),
                        "raw_solution": solution,
                    }
                )
                + "\n"
            )


def generate(args: argparse.Namespace) -> None:
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world-size)")
    if not args.model:
        raise ValueError("--model is required unless --merge is used")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_keys(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    tasks = list(load_dataset(args.dataset).items())

    requests = []
    for task_index, (task_id, task) in enumerate(tasks):
        if task_index % args.world_size != args.rank:
            continue
        prompt, direct_completion = build_prompt(
            task, tokenizer, args.enable_thinking, args.prompt_contract
        )
        for sample_id in range(args.n_samples):
            if (task_id, sample_id) not in completed:
                requests.append(
                    {
                        "task_id": task_id,
                        "sample_id": sample_id,
                        "entry_point": task["entry_point"],
                        "prompt": prompt,
                        "direct_completion": direct_completion,
                    }
                )

    print(
        f"rank={args.rank}: {len(requests)} pending requests "
        f"across {len(tasks)} {args.dataset} tasks",
        flush=True,
    )
    if not requests:
        return

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_num_batched_tokens=65536,
        max_num_seqs=64,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    for start in range(0, len(requests), args.request_chunk_size):
        chunk = requests[start : start + args.request_chunk_size]
        outputs = llm.generate([item["prompt"] for item in chunk], sampling_params, use_tqdm=True)
        write_records(args.output, chunk, outputs)
        print(f"rank={args.rank}: wrote {start + len(chunk)}/{len(requests)} requests", flush=True)


def merge(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    for rank in range(args.world_size):
        shard = args.output.parent / f"{args.dataset}_rank{rank}.jsonl"
        if not shard.exists():
            raise FileNotFoundError(f"missing shard: {shard}")
        with shard.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[(row["task_id"], int(row["sample_id"]))] = row

    expected = len(load_dataset(args.dataset)) * args.n_samples
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} samples, found {len(rows)}")
    with args.output.open("w") as handle:
        for _, row in sorted(rows.items()):
            handle.write(json.dumps(row) + "\n")
    print(f"merged {len(rows)} {args.dataset} samples into {args.output}", flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.merge:
        merge(arguments)
    else:
        generate(arguments)
