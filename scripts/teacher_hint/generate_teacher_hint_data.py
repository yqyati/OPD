#!/usr/bin/env python3
"""Generate direct teacher hints and attach them as assistant-prefix targets.

Hint generation uses a dedicated meta-prompt, but that prompt is never copied
into training data. Training reconstructs the original problem prompt and uses
only the normalized hint as an assistant prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "data_filter"))


GUIDE_INSTRUCTION = (
    "Help a smaller model solve the problem. Think through it internally, then write one concise and actionable "
    "hint inside <HINT>...</HINT>. Point to the key idea and the first useful step, without giving the full "
    "solution or final answer."
)

ANSWER_LEAK_PATTERN = re.compile(r"\\boxed\s*\{|(?:final\s+)?answer\s+is", re.IGNORECASE)


def extract_prompt_text(row: pd.Series) -> str:
    """Extract the problem text without importing the torch-based scorer.

    Hint-data generation only needs dataframe normalization and vLLM.  Keeping
    this lightweight helper local avoids importing score_prompt_opd_data at
    module load time, which would require torch before generation starts.
    """
    if "prompt" in row and row["prompt"] is not None:
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, (list, tuple)) and prompt:
            item = prompt[0]
            if isinstance(item, dict) and "content" in item:
                return str(item["content"]).strip()
        if isinstance(prompt, str):
            return prompt.strip()
    for column in ("problem", "question"):
        if column in row and row[column] is not None:
            return str(row[column]).strip()
    raise ValueError("Cannot find prompt/problem/question column in row.")


def normalize_hint(text: str) -> str:
    raw = (text or "").strip()
    # Accept only a complete tagged block.  Untagged thinking or a truncated
    # generation must never be silently converted into a training hint.
    match = re.search(r"<HINT>\s*(.*?)\s*</HINT>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return ""
    hint = " ".join(match.group(1).strip().split()).strip('"')
    changed = True
    while changed:
        changed = False
        for prefix in ("Assistant:", "Strategy:", "Hint:", "Guide:"):
            if hint.lower().startswith(prefix.lower()):
                hint = hint[len(prefix) :].strip()
                changed = True
    return hint.strip()


def select_hint_candidate(generated_candidates) -> tuple[object, str, bool]:
    ranked = []
    for candidate_idx, generated in enumerate(generated_candidates):
        hint = normalize_hint(generated.text)
        finish_reason = str(generated.finish_reason)
        valid = (
            bool(hint)
            and len(hint.split()) >= 4
            and "length" not in finish_reason.lower()
            and ANSWER_LEAK_PATTERN.search(hint) is None
        )
        ranked.append((not valid, candidate_idx, len(generated.token_ids), generated, hint, valid))
    _, _, _, generated, hint, valid = min(ranked, key=lambda item: (item[0], item[1], item[2]))
    return generated, hint, valid


def build_retry_guide_chat(row: pd.Series) -> list[dict]:
    problem = extract_problem(row)
    retry_instruction = (
        "Return exactly one block and nothing else: <HINT>one concise actionable hint</HINT>. "
        "Do not include a final answer, boxed answer, or any text outside the HINT tags."
    )
    return [
        {"role": "system", "content": retry_instruction},
        {"role": "user", "content": problem},
    ]


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
    parser.add_argument("--num-candidates", type=int, default=1)
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
    if args.force:
        # A previous interrupted run may have a partial temp file whose rows
        # were generated with older parsing rules.  Force mode must regenerate
        # those rows rather than treating them as completed.
        if temp_jsonl.exists():
            temp_jsonl.unlink()
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
            n=args.num_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            # Do not stop on </HINT>: vLLM removes the stop string from the
            # returned text, which would make strict tagged-block parsing fail.
            stop=None,
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
                    generated, guide_text, candidate_valid = select_hint_candidate(output.outputs)
                    retry_count = 0
                    invalid_reason = ""
                    if not candidate_valid:
                        retry_count = 1
                        retry_prompt = tokenizer.apply_chat_template(
                            build_retry_guide_chat(df.iloc[idx]),
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        retry_output = llm.generate([retry_prompt], sampling)[0]
                        generated, guide_text, candidate_valid = select_hint_candidate(retry_output.outputs)
                        if not candidate_valid:
                            invalid_reason = "retry_failed_plain_fallback"

                    # A failed retry is deliberately a plain-OPD sample: keep
                    # the row but provide no teacher prefix at all.
                    teacher_prefix_text = f"<HINT>\n{guide_text}\n</HINT>\n" if candidate_valid else ""
                    teacher_prefix_token_ids = tokenizer.encode(
                        teacher_prefix_text, add_special_tokens=False
                    )
                    record = {
                        "__teacher_guide_row_id": idx,
                        "teacher_guide_text": guide_text,
                        "teacher_prefix_text": teacher_prefix_text,
                        "teacher_prefix_token_ids": [int(token_id) for token_id in teacher_prefix_token_ids],
                        "teacher_prefix_token_len": len(teacher_prefix_token_ids),
                        "teacher_guide_token_len": len(generated.token_ids),
                        "teacher_guide_finish_reason": str(generated.finish_reason),
                        "teacher_guide_model": args.teacher_model,
                        "teacher_guide_max_tokens": args.max_new_tokens,
                        "teacher_guide_temperature": args.temperature,
                        "teacher_guide_top_p": args.top_p,
                        "teacher_hint_prompt_type": "direct_hint_v3_tagged_four_candidates_separate_generation_prompt",
                        "teacher_hint_candidate_valid": candidate_valid,
                        "teacher_hint_num_candidates": args.num_candidates,
                        "teacher_hint_retry_count": retry_count,
                        "teacher_hint_invalid_reason": invalid_reason,
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
    out_df["teacher_prefix_text"] = out_df["teacher_prefix_text"].fillna("")
    out_df = out_df.drop(columns=["__teacher_guide_row_id"])
    out_df.to_parquet(output_path, index=False)
    valid_count = int(out_df["teacher_hint_candidate_valid"].fillna(False).astype(bool).sum())
    plain_fallback_count = len(out_df) - valid_count
    retry_count = int((out_df["teacher_hint_retry_count"].fillna(0) > 0).sum())
    print(
        f"Wrote {output_path} ({len(out_df)} rows); valid_hints={valid_count}; "
        f"retry_attempted={retry_count}; plain_fallback={plain_fallback_count}"
    )


if __name__ == "__main__":
    main()
