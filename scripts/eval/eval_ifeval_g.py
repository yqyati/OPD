#!/usr/bin/env python3
"""Resumable Qwen3 generation and IFEvalG scoring for held-out IFEval.

Generation is deliberately separate per GPU rank.  The score mode merges those
rank files, removes Qwen thinking text, and checks every official IFEval
constraint with the same IFEvalG registry used by IF-RLVR training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OPEN_INSTRUCT = ROOT / "third_party" / "open-instruct-ifrlvr"
if str(OPEN_INSTRUCT) not in sys.path:
    sys.path.insert(0, str(OPEN_INSTRUCT))

from open_instruct.IFEvalG import instructions_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "score"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=7168)
    parser.add_argument("--max-model-len", type=int, default=9216)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--request-chunk-size", type=int, default=32)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    required = {"key", "prompt", "instruction_id_list", "kwargs"}
    if not rows or any(required.difference(row) for row in rows):
        raise ValueError(f"Invalid IFEval input: {path}")
    keys = [row["key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("IFEval input contains duplicate keys")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def visible_answer(completion: str) -> str:
    answer = completion.replace("<|assistant|>", "").strip()
    answer = answer.split("</think>")[-1]
    return answer.replace("<answer>", "").replace("</answer>", "").strip()


def generate(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("--model is required in generate mode")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world-size)")
    rows = load_rows(args.input)
    output = args.output_dir / f"generations_rank{args.rank}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {row["key"] for row in read_jsonl(output)}
    owned = [row for index, row in enumerate(rows) if index % args.world_size == args.rank]
    pending = [row for row in owned if row["key"] not in completed]
    print(f"rank={args.rank}: owned={len(owned)}, completed={len(completed)}, pending={len(pending)}", flush=True)
    if not pending:
        return

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=16384,
        max_num_seqs=args.request_chunk_size,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    with output.open("a") as handle:
        for start in range(0, len(pending), args.request_chunk_size):
            chunk = pending[start : start + args.request_chunk_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                for row in chunk
            ]
            outputs = llm.generate(prompts, sampling, use_tqdm=True)
            for row, result in zip(chunk, outputs, strict=True):
                choice = result.outputs[0]
                handle.write(
                    json.dumps(
                        {
                            "key": row["key"],
                            "prompt": row["prompt"],
                            "instruction_id_list": row["instruction_id_list"],
                            "kwargs": row["kwargs"],
                            "response": choice.text,
                            "finish_reason": choice.finish_reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            print(f"rank={args.rank}: wrote {start + len(chunk)}/{len(pending)}", flush=True)


def check_constraint(instruction_id: str, kwargs: dict[str, Any] | None, answer: str) -> bool:
    arguments = {} if kwargs is None else {key: value for key, value in kwargs.items() if value is not None}
    checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    checker.build_description(**arguments)
    return bool(checker.check_following(answer))


def score(args: argparse.Namespace) -> None:
    rows = load_rows(args.input)
    expected = {row["key"] for row in rows}
    generations: dict[int, dict[str, Any]] = {}
    for rank in range(args.world_size):
        path = args.output_dir / f"generations_rank{rank}.jsonl"
        for row in read_jsonl(path):
            key = row["key"]
            if key in generations:
                raise ValueError(f"Duplicate generated key {key} across rank files")
            generations[key] = row
    missing = sorted(expected.difference(generations))
    extra = sorted(set(generations).difference(expected))
    if missing or extra:
        raise ValueError(f"Incomplete generation: missing={len(missing)}, extra={len(extra)}")

    scored: list[dict[str, Any]] = []
    per_type: dict[str, list[bool]] = defaultdict(list)
    total_constraints = 0
    passed_constraints = 0
    prompt_all_pass = 0
    prompt_by_count: dict[int, list[bool]] = defaultdict(list)
    for item in rows:
        generated = generations[item["key"]]
        answer = visible_answer(generated["response"])
        checks: list[dict[str, Any]] = []
        for instruction_id, kwargs in zip(item["instruction_id_list"], item["kwargs"], strict=True):
            try:
                passed = check_constraint(instruction_id, kwargs, answer)
                error = None
            except Exception as exc:  # preserve isolated verifier failures for auditing
                passed = False
                error = type(exc).__name__
            checks.append({"instruction_id": instruction_id, "passed": passed, "verifier_error": error})
            per_type[instruction_id].append(passed)
            total_constraints += 1
            passed_constraints += int(passed)
        all_pass = bool(checks) and all(check["passed"] for check in checks)
        prompt_all_pass += int(all_pass)
        prompt_by_count[len(checks)].append(all_pass)
        scored.append({**generated, "visible_answer": answer, "checks": checks, "all_constraints_passed": all_pass})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = args.output_dir / "generations_merged.jsonl"
    with merged.open("w") as handle:
        for row in sorted(generations.values(), key=lambda row: row["key"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    scored_path = args.output_dir / "scored.jsonl"
    with scored_path.open("w") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "benchmark": "google/IFEval",
        "num_prompts": len(rows),
        "num_constraints": total_constraints,
        "constraint_level_accuracy": passed_constraints / total_constraints,
        "prompt_level_all_constraints_accuracy": prompt_all_pass / len(rows),
        "by_constraint_count": {
            str(count): {"num_prompts": len(values), "all_constraints_accuracy": sum(values) / len(values)}
            for count, values in sorted(prompt_by_count.items())
        },
        "by_instruction_type": {
            instruction_id: {"num_constraints": len(values), "accuracy": sum(values) / len(values)}
            for instruction_id, values in sorted(per_type.items())
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.mode == "generate":
        generate(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
