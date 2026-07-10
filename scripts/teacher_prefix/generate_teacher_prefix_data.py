#!/usr/bin/env python3
"""Generate teacher prefixes and attach them to an OPD training parquet.

The generated prefix is appended after the chat template's generation prompt at
training time. It is stored as plain text so the existing RL dataset can treat it
as prompt context and keep OPD/reward loss on the student-generated suffix only.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input training parquet.")
    parser.add_argument("--output", required=True, help="Output parquet with teacher_prefix_text attached.")
    parser.add_argument("--teacher-model", required=True, help="Teacher model path used for prefix generation.")
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated visible GPU ids.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--top-k-logprobs",
        type=int,
        default=0,
        help="If >0, ask vLLM to return top-k logprobs for each generated prefix token and store them.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows for smoke tests.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output parquet.")
    return parser.parse_args()


def extract_prompt_chat(row: pd.Series) -> list[dict]:
    prompt = row.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)) and prompt:
        return list(prompt)
    return [{"role": "user", "content": extract_prompt_text(row)}]


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
            idx = record.get("__teacher_prefix_row_id", record.get("__opd_original_index"))
            if isinstance(idx, int):
                completed[idx] = record
    return completed


def extract_topk_logprobs(generated, top_k: int) -> tuple[list[list[int]], list[list[float]]]:
    """Extract vLLM per-step top-k logprobs into JSON-serializable lists."""
    if top_k <= 0:
        return [], []
    logprobs = getattr(generated, "logprobs", None)
    if not logprobs:
        return [], []

    all_ids: list[list[int]] = []
    all_logp: list[list[float]] = []
    for step_logprobs in logprobs:
        if not step_logprobs:
            all_ids.append([])
            all_logp.append([])
            continue
        entries = []
        for token_id, value in step_logprobs.items():
            logprob = getattr(value, "logprob", value)
            entries.append((int(token_id), float(logprob)))
        entries.sort(key=lambda x: x[1], reverse=True)
        entries = entries[:top_k]
        all_ids.append([token_id for token_id, _ in entries])
        all_logp.append([logprob for _, logprob in entries])
    return all_ids, all_logp


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
    df["__teacher_prefix_row_id"] = range(len(df))
    if "__opd_original_index" not in df.columns:
        df["__opd_original_index"] = range(len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_jsonl = output_path.with_suffix(".jsonl.tmp")
    completed = load_completed(temp_jsonl)
    pending = [i for i in range(len(df)) if i not in completed]
    print(f"Loaded {len(df)} prompts. Pending teacher-prefix generation: {len(pending)}")

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
        sampling_kwargs = dict(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        if args.top_k_logprobs > 0:
            sampling_kwargs["logprobs"] = args.top_k_logprobs
        sampling = SamplingParams(**sampling_kwargs)

        with temp_jsonl.open("a", encoding="utf-8") as f_out:
            for start in range(0, len(pending), args.batch_size):
                batch_indices = pending[start : start + args.batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        extract_prompt_chat(df.iloc[idx]),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                    for idx in batch_indices
                ]
                outputs = llm.generate(prompts, sampling)
                for idx, output in zip(batch_indices, outputs, strict=True):
                    generated = output.outputs[0]
                    topk_ids, topk_logp = extract_topk_logprobs(generated, args.top_k_logprobs)
                    record = {
                        "__teacher_prefix_row_id": idx,
                        "teacher_prefix_text": generated.text,
                        "teacher_prefix_token_len": len(generated.token_ids),
                        "teacher_prefix_finish_reason": str(generated.finish_reason),
                        "teacher_prefix_model": args.teacher_model,
                        "teacher_prefix_max_tokens": args.max_new_tokens,
                        "teacher_prefix_temperature": args.temperature,
                        "teacher_prefix_top_p": args.top_p,
                    }
                    if args.top_k_logprobs > 0:
                        record["teacher_prefix_top_k_ids"] = topk_ids
                        record["teacher_prefix_top_k_log_probs"] = topk_logp
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                done = min(start + len(batch_indices), len(pending))
                print(f"Generated {done}/{len(pending)} pending teacher prefixes")

    completed = load_completed(temp_jsonl)
    if len(completed) != len(df):
        missing = len(df) - len(completed)
        raise RuntimeError(f"Teacher-prefix generation incomplete: missing {missing} rows.")

    prefix_df = pd.DataFrame([completed[i] for i in range(len(df))])
    if "__opd_original_index" in prefix_df.columns:
        prefix_df = prefix_df.drop(columns=["__opd_original_index"])
    out_df = df.merge(prefix_df, on="__teacher_prefix_row_id", how="left")
    if out_df["teacher_prefix_text"].isna().any():
        raise RuntimeError("Merged output contains empty teacher_prefix_text rows.")
    out_df = out_df.drop(columns=["__teacher_prefix_row_id"])
    out_df.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
