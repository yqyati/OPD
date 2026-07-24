#!/usr/bin/env python3
"""Annotate free-form handoff boundaries on immutable teacher trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = "You annotate solution trajectories. Return only the requested JSON object."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional deterministic prefix of input rows for sanity checks.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [{"role": "user", "content": value}]
    if isinstance(value, list):
        messages = []
        for message in value:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
                    )
                messages.append({"role": str(message.get("role", "user")), "content": str(content)})
        if messages:
            return messages
    return [{"role": "user", "content": str(value)}]


def problem_text(row: pd.Series) -> str:
    if "prompt" in row and row["prompt"] is not None:
        return "\n".join(f"{message['role']}: {message['content']}" for message in normalize_messages(row["prompt"]))
    return str(row.get("question", ""))


def teacher_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(token_id) for token_id in value]


def annotation_user_prompt(problem: str, trace: str, retry: bool = False) -> str:
    retry_note = "Your previous response was invalid. " if retry else ""
    return f"""{retry_note}Below is an original, already-generated solution trace for a math problem.

Problem:
{problem}

Original trace:
{trace}

Identify the earliest handoff position at which another capable solver has a specific, problem-dependent plan to execute.
Before the handoff, the trace must contain a concrete reduction, construction, invariant, case decomposition, or explicit problem-specific transformation.
Before the handoff, the trace must NOT contain numerical evaluation of that plan, exhaustive case work, Python/code execution, a final value, a boxed answer, or a final conclusion.

Return a short, exact, contiguous anchor from the ORIGINAL TRACE: 8 to 20 words that end exactly at the handoff position.
Do not paraphrase. Do not return a formula-only anchor, a numeric answer, or a final-conclusion sentence.
Return exactly one JSON object and nothing else:
{{"handoff_anchor": "<8-20 verbatim words ending at the handoff>"}}"""


def parse_anchor(text: str, trace: str) -> tuple[str, int] | None:
    decoder = json.JSONDecoder()
    for start in (idx for idx, char in enumerate(text) if char == "{"):
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        anchor = payload.get("handoff_anchor") if isinstance(payload, dict) else None
        if not isinstance(anchor, str):
            continue
        words = anchor.strip().split()
        if not 8 <= len(words) <= 20:
            continue
        first = trace.find(anchor)
        if first >= 0 and trace.find(anchor, first + 1) < 0:
            return anchor, first + len(anchor)
    return None


def char_offset_to_token_budget(tokenizer, token_ids: list[int], char_offset: int) -> int:
    """Map a decoded-text boundary back to the first generated-token boundary at/after it."""
    low, high = 1, len(token_ids)
    while low < high:
        mid = (low + high) // 2
        if len(tokenizer.decode(token_ids[:mid], skip_special_tokens=False)) < char_offset:
            low = mid + 1
        else:
            high = mid
    return low


def main() -> None:
    args = parse_args()

    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")

    frame = pd.read_parquet(args.input)
    required = {"teacher_prefix_token_ids", "prompt"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"input is missing required columns: {sorted(missing)}")
    if args.max_rows > 0:
        frame = frame.iloc[: args.max_rows].copy()

    frame = frame.iloc[args.shard_id :: args.num_shards].copy()
    frame["teacher_handoff_annotation_row_id"] = frame.index
    frame.reset_index(drop=True, inplace=True)
    print(f"annotating shard {args.shard_id}/{args.num_shards}: rows={len(frame)}")

    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=128, logprobs=0)

    records: list[dict[str, Any]] = [
        {"budget": None, "status": "pending", "attempts": 0, "raw": "", "quote": ""} for _ in range(len(frame))
    ]
    pending = list(range(len(frame)))
    for attempt in range(args.max_retries + 1):
        if not pending:
            break
        requests = []
        request_indices = []
        for row_idx in pending:
            row = frame.iloc[row_idx]
            token_ids = teacher_token_ids(row["teacher_prefix_token_ids"])
            if not token_ids:
                records[row_idx] = {"budget": 0, "status": "empty_trace", "attempts": attempt, "raw": "", "quote": ""}
                continue
            trace = tokenizer.decode(token_ids, skip_special_tokens=False)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": annotation_user_prompt(problem_text(row), trace, retry=attempt > 0)},
            ]
            requests.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            request_indices.append(row_idx)

        next_pending = []
        for start in range(0, len(requests), args.batch_size):
            request_batch = requests[start : start + args.batch_size]
            output_batch = llm.generate(request_batch, sampling)
            for row_idx, generated in zip(request_indices[start : start + args.batch_size], output_batch, strict=True):
                raw = generated.outputs[0].text
                token_ids = teacher_token_ids(frame.iloc[row_idx]["teacher_prefix_token_ids"])
                trace = tokenizer.decode(token_ids, skip_special_tokens=False)
                parsed = parse_anchor(raw, trace)
                if parsed is None:
                    records[row_idx] = {
                        "budget": None,
                        "status": "invalid_json_or_anchor",
                        "attempts": attempt + 1,
                        "raw": raw,
                        "quote": "",
                    }
                    next_pending.append(row_idx)
                else:
                    anchor, char_offset = parsed
                    records[row_idx] = {
                        "budget": char_offset_to_token_budget(tokenizer, token_ids, char_offset),
                        "status": "ok",
                        "attempts": attempt + 1,
                        "raw": raw,
                        "quote": anchor,
                    }
        pending = next_pending
        print(f"annotation attempt {attempt + 1}: pending={len(pending)} / {len(frame)}")

    frame["teacher_handoff_token_budget"] = [record["budget"] for record in records]
    frame["teacher_handoff_annotation_status"] = [record["status"] for record in records]
    frame["teacher_handoff_annotation_attempts"] = [record["attempts"] for record in records]
    frame["teacher_handoff_annotation_raw"] = [record["raw"] for record in records]
    frame["teacher_handoff_terminal_quote"] = [record["quote"] for record in records]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)

    ok = frame[frame["teacher_handoff_annotation_status"] == "ok"]
    stats = {
        "rows": len(frame),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "ok": len(ok),
        "parse_rate": len(ok) / len(frame) if len(frame) else 0.0,
        "budget_quantiles": ok["teacher_handoff_token_budget"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
        "budget_counts": ok["teacher_handoff_token_budget"].value_counts().sort_index().to_dict(),
    }
    stats_path = output.with_suffix(".summary.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
