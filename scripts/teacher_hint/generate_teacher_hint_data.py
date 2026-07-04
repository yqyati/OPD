#!/usr/bin/env python3
"""Generate short teacher guides and attach them as teacher_prefix_text.

This intentionally reuses the existing teacher-prefix training path. The
generated text is a concise guide instead of a raw teacher CoT continuation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "data_filter"))

from score_prompt_opd_data import extract_prompt_text


GUIDE_INSTRUCTION = (
    "You are a math strategy teacher. Given the problem, write a concise guide for solving it. "
    "Do not solve the problem completely. Do not compute or reveal the final answer. "
    "Do not write a long chain-of-thought. Keep the guide within 1-3 sentences."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet.")
    parser.add_argument("--output", required=True, help="Output parquet with guide-style teacher_prefix_text attached.")
    parser.add_argument("--teacher-model", required=True, help="Teacher model path used for guide generation.")
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated visible GPU ids.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows for smoke tests.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output parquet.")
    return parser.parse_args()


def extract_problem(row: pd.Series) -> str:
    if "problem" in row and isinstance(row.get("problem"), str):
        return row["problem"].strip()
    return extract_prompt_text(row).strip()


def build_guide_chat(row: pd.Series) -> list[dict]:
    problem = extract_problem(row)
    return [
        {"role": "system", "content": GUIDE_INSTRUCTION},
        {"role": "user", "content": problem},
    ]


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
            idx = record.get("__teacher_guide_row_id")
            if isinstance(idx, int):
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
    df["__teacher_guide_row_id"] = range(len(df))
    if "__opd_original_index" not in df.columns:
        df["__opd_original_index"] = range(len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_jsonl = output_path.with_suffix(".jsonl.tmp")
    completed = load_completed(temp_jsonl)
    pending = [i for i in range(len(df)) if i not in completed]
    print(f"Loaded {len(df)} prompts. Pending teacher-guide generation: {len(pending)}")

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
                prompts = [
                    tokenizer.apply_chat_template(
                        build_guide_chat(df.iloc[idx]),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                    for idx in batch_indices
                ]
                outputs = llm.generate(prompts, sampling)
                for idx, output in zip(batch_indices, outputs, strict=True):
                    generated = output.outputs[0]
                    guide_text = generated.text.strip()
                    teacher_prefix_text = f"\n\nGuide:\n{guide_text}\n\nNow solve the problem.\n"
                    record = {
                        "__teacher_guide_row_id": idx,
                        "teacher_guide_text": guide_text,
                        "teacher_prefix_text": teacher_prefix_text,
                        "teacher_guide_token_len": len(generated.token_ids),
                        "teacher_guide_finish_reason": str(generated.finish_reason),
                        "teacher_guide_model": args.teacher_model,
                        "teacher_guide_max_tokens": args.max_new_tokens,
                        "teacher_guide_temperature": args.temperature,
                        "teacher_guide_top_p": args.top_p,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                done = min(start + len(batch_indices), len(pending))
                print(f"Generated {done}/{len(pending)} pending teacher guides")

    completed = load_completed(temp_jsonl)
    if len(completed) != len(df):
        missing = len(df) - len(completed)
        raise RuntimeError(f"Teacher-guide generation incomplete: missing {missing} rows.")

    guide_df = pd.DataFrame([completed[i] for i in range(len(df))])
    out_df = df.merge(guide_df, on="__teacher_guide_row_id", how="left")
    if out_df["teacher_prefix_text"].isna().any():
        raise RuntimeError("Merged output contains empty teacher_prefix_text rows.")
    out_df = out_df.drop(columns=["__teacher_guide_row_id"])
    out_df.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
