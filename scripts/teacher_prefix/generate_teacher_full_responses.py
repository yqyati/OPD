#!/usr/bin/env python3
"""Generate resumable, exact-token full teacher responses for OPD prefixes.

The output is deliberately a teacher-response asset rather than a single
prefix dataset.  It stores every generated token ID through the model's native
EOS (or the requested maximum), so fixed or dynamic prefix datasets can later
be sliced from the *same sampled trajectory* without rolling out the teacher
again.
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
    parser.add_argument("--input", required=True, help="Input OPD training parquet.")
    parser.add_argument("--output", required=True, help="Output full teacher-response parquet.")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output parquet.")
    return parser.parse_args()


def extract_prompt_chat(row: pd.Series) -> list[dict]:
    prompt = row.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)) and prompt:
        return list(prompt)
    return [{"role": "user", "content": extract_prompt_text(row)}]


def load_completed(path: Path) -> dict[int, dict]:
    completed: dict[int, dict] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                row_id = record["__teacher_response_row_id"]
                if isinstance(row_id, int):
                    completed[row_id] = record
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
    return completed


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_model_len < args.max_new_tokens:
        raise ValueError("--max-model-len must cover --max-new-tokens plus the prompt")

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"{output_path} already exists; reusing it (pass --force to regenerate).")
        return

    if output_path.exists():
        output_path.unlink()
    temp_jsonl = output_path.with_suffix(".jsonl.tmp")
    if args.force and temp_jsonl.exists():
        temp_jsonl.unlink()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    dataframe = pd.read_parquet(args.input).reset_index(drop=True)
    if args.limit is not None:
        dataframe = dataframe.head(args.limit).copy()
    dataframe["__teacher_response_row_id"] = range(len(dataframe))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(temp_jsonl)
    pending = [row_id for row_id in range(len(dataframe)) if row_id not in completed]
    print(f"Loaded {len(dataframe)} prompts. Pending full teacher responses: {len(pending)}")

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
        # Do not set ignore_eos=True: natural EOS must terminate that trace.
        sampling = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )

        with temp_jsonl.open("a", encoding="utf-8") as stream:
            for start in range(0, len(pending), args.batch_size):
                row_ids = pending[start : start + args.batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        extract_prompt_chat(dataframe.iloc[row_id]),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                    for row_id in row_ids
                ]
                outputs = llm.generate(prompts, sampling)
                for row_id, output in zip(row_ids, outputs, strict=True):
                    generated = output.outputs[0]
                    stream.write(
                        json.dumps(
                            {
                                "__teacher_response_row_id": row_id,
                                "teacher_response_text": generated.text,
                                "teacher_response_token_ids": [int(x) for x in generated.token_ids],
                                "teacher_response_token_len": len(generated.token_ids),
                                "teacher_response_finish_reason": str(generated.finish_reason),
                                "teacher_response_model": args.teacher_model,
                                "teacher_response_max_tokens": args.max_new_tokens,
                                "teacher_response_temperature": args.temperature,
                                "teacher_response_top_p": args.top_p,
                                "teacher_response_enable_thinking": bool(args.enable_thinking),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                stream.flush()
                print(f"Generated {min(start + len(row_ids), len(pending))}/{len(pending)} pending responses")

    completed = load_completed(temp_jsonl)
    if len(completed) != len(dataframe):
        raise RuntimeError(f"Full-response generation incomplete: {len(dataframe) - len(completed)} rows missing")

    response_columns = pd.DataFrame([completed[row_id] for row_id in range(len(dataframe))])
    output = dataframe.merge(response_columns, on="__teacher_response_row_id", how="left")
    output = output.drop(columns=["__teacher_response_row_id"])
    if output["teacher_response_token_ids"].isna().any():
        raise RuntimeError("Merged output has missing teacher responses")
    output.to_parquet(output_path, index=False)
    print(f"Wrote {output_path}: rows={len(output)}")


if __name__ == "__main__":
    main()
