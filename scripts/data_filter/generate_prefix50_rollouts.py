#!/usr/bin/env python3
"""Generate short student rollouts for OPD prefix-based filtering."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from score_prompt_opd_data import extract_prompt_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input training parquet.")
    parser.add_argument("--output", required=True, help="Output parquet with generated prefixes.")
    parser.add_argument("--model", required=True, help="Student model path used for generation.")
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated visible GPU ids.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows for smoke tests.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output parquet.")
    return parser.parse_args()


def extract_prompt_chat(row: pd.Series):
    prompt = row.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)) and prompt:
        return list(prompt)
    return [{"role": "user", "content": extract_prompt_text(row)}]


def load_completed(temp_jsonl: Path) -> dict[int, dict]:
    completed = {}
    if not temp_jsonl.exists():
        return completed
    with temp_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = record.get("__opd_original_index")
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
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.input).reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    df["__opd_original_index"] = range(len(df))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_jsonl = output_path.with_suffix(".jsonl.tmp")
    completed = load_completed(temp_jsonl)

    pending = [i for i in range(len(df)) if i not in completed]
    print(f"Loaded {len(df)} prompts. Pending prefix generation: {len(pending)}")

    llm = LLM(
        model=args.model,
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
            prompts = []
            for idx in batch_indices:
                chat = extract_prompt_chat(df.iloc[idx])
                prompts.append(
                    tokenizer.apply_chat_template(
                        chat,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                )
            outputs = llm.generate(prompts, sampling)
            for idx, output in zip(batch_indices, outputs):
                text = output.outputs[0].text
                record = {
                    "__opd_original_index": idx,
                    "prefix50_text": text,
                    "prefix50_finish_reason": str(output.outputs[0].finish_reason),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            done = min(start + len(batch_indices), len(pending))
            print(f"Generated {done}/{len(pending)} pending prefixes")

    completed = load_completed(temp_jsonl)
    rows = [completed[i] for i in sorted(completed)]
    prefix_df = pd.DataFrame(rows)
    if len(prefix_df) != len(df):
        missing = len(df) - len(prefix_df)
        raise RuntimeError(f"Prefix generation incomplete: missing {missing} rows.")
    prefix_df.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(prefix_df)} rows)")


if __name__ == "__main__":
    main()
