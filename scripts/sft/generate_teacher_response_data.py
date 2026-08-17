#!/usr/bin/env python3
"""Generate full teacher responses and attach them to a training parquet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=7168)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--skip-overlong-prompts",
        action="store_true",
        help="Record prompts exceeding --max-model-len as skipped and continue the resumable run.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_prompt(prompt) -> list[dict]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"Expected non-empty prompt list, got {type(prompt)}")
    return [dict(m) for m in prompt]


def load_completed(temp_jsonl: Path) -> dict[int, dict]:
    completed: dict[int, dict] = {}
    if not temp_jsonl.exists():
        return completed
    with temp_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = record.get("__teacher_response_row_id")
            if isinstance(idx, int):
                # Older resumable records predate these fields. Normalize them
                # on read so mixed-version JSONL always materializes one schema.
                record.setdefault("teacher_response_status", "generated")
                record.setdefault("teacher_response_prompt_token_len", None)
                completed[idx] = record
    return completed


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"{output_path} already exists. Use --force to regenerate.")
        return

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    df = pd.read_parquet(args.input).reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    df["__teacher_response_row_id"] = range(len(df))
    if "__opd_original_index" not in df.columns:
        df["__opd_original_index"] = range(len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_jsonl = output_path.with_suffix(".jsonl.tmp")
    completed = load_completed(temp_jsonl)
    pending = [i for i in range(len(df)) if i not in completed]
    print(f"Loaded {len(df)} prompts. Pending teacher-response generation: {len(pending)}")

    if pending:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.teacher_model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )

        with temp_jsonl.open("a", encoding="utf-8") as f_out:
            for start in range(0, len(pending), args.batch_size):
                batch_indices = pending[start : start + args.batch_size]
                rendered_prompts = {
                    idx: tokenizer.apply_chat_template(
                        normalize_prompt(df.iloc[idx]["prompt"]),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                    for idx in batch_indices
                }
                prompt_lengths = {
                    idx: len(tokenizer.encode(prompt, add_special_tokens=False))
                    for idx, prompt in rendered_prompts.items()
                }
                overlong = {
                    idx: prompt_len
                    for idx, prompt_len in prompt_lengths.items()
                    if prompt_len > args.max_model_len
                }
                if overlong and not args.skip_overlong_prompts:
                    idx, prompt_len = next(iter(overlong.items()))
                    raise ValueError(
                        f"Rendered prompt at row {idx} has {prompt_len} tokens, which exceeds "
                        f"--max-model-len={args.max_model_len}."
                    )

                valid_indices = [idx for idx in batch_indices if idx not in overlong]
                outputs = (
                    llm.generate([rendered_prompts[idx] for idx in valid_indices], sampling)
                    if valid_indices
                    else []
                )
                generated_by_index = dict(zip(valid_indices, outputs, strict=True))
                for idx in batch_indices:
                    if idx in overlong:
                        record = {
                            "__teacher_response_row_id": idx,
                            "teacher_response_text": "",
                            "teacher_response_token_ids": [],
                            "teacher_response_token_len": 0,
                            "teacher_response_finish_reason": "prompt_too_long",
                            "teacher_response_status": "skipped_prompt_too_long",
                            "teacher_response_prompt_token_len": overlong[idx],
                            "teacher_response_model": args.teacher_model,
                            "teacher_response_max_tokens": args.max_new_tokens,
                            "teacher_response_temperature": args.temperature,
                            "teacher_response_top_p": args.top_p,
                            "teacher_response_enable_thinking": bool(args.enable_thinking),
                        }
                        print(
                            f"Skipped overlong prompt row={idx} tokens={overlong[idx]} "
                            f"max_model_len={args.max_model_len}"
                        )
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        continue

                    output = generated_by_index[idx]
                    generated = output.outputs[0]
                    record = {
                        "__teacher_response_row_id": idx,
                        "teacher_response_text": generated.text,
                        "teacher_response_token_ids": [int(token_id) for token_id in generated.token_ids],
                        "teacher_response_token_len": len(generated.token_ids),
                        "teacher_response_finish_reason": str(generated.finish_reason),
                        "teacher_response_status": "generated",
                        "teacher_response_model": args.teacher_model,
                        "teacher_response_max_tokens": args.max_new_tokens,
                        "teacher_response_temperature": args.temperature,
                        "teacher_response_top_p": args.top_p,
                        "teacher_response_enable_thinking": bool(args.enable_thinking),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                done = min(start + len(batch_indices), len(pending))
                print(f"Generated {done}/{len(pending)} pending teacher responses")

    completed = load_completed(temp_jsonl)
    if len(completed) != len(df):
        missing = len(df) - len(completed)
        raise RuntimeError(f"Teacher-response generation incomplete: missing {missing} rows.")

    response_records = [completed[i] for i in range(len(df))]
    if any(
        record.get("teacher_response_status", "generated") == "generated"
        and not record.get("teacher_response_text")
        for record in response_records
    ):
        raise RuntimeError("Merged output contains empty teacher_response_text rows.")
    # Build the final parquet with Arrow rather than pandas merge.  Eurus rows
    # contain nested prompt/reward columns; pandas can silently drop the
    # object-valued token-id list when materializing those columns together.
    source_table = parquet.read_table(args.input)
    source_names = set(source_table.column_names)
    source_table = source_table.drop(["__teacher_response_row_id"] if "__teacher_response_row_id" in source_names else [])
    response_table = pa.Table.from_pylist(
        [{key: value for key, value in record.items() if key != "__teacher_response_row_id"} for record in response_records]
    )
    response_names = set(response_table.column_names)
    duplicate_names = source_names.intersection(response_names)
    if duplicate_names:
        raise RuntimeError(f"Unexpected source/response column collision: {sorted(duplicate_names)}")
    out_table = pa.Table.from_arrays(
        list(source_table.columns) + list(response_table.columns),
        names=source_table.column_names + response_table.column_names,
    )
    parquet.write_table(out_table, output_path, compression="zstd")
    print(f"Wrote {output_path} ({out_table.num_rows} rows)")


if __name__ == "__main__":
    main()
