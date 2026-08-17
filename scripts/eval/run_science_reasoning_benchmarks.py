#!/usr/bin/env python3
"""Rule-based single-sample evaluation on GPQA-Diamond and SciBench."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


ROOT = Path(__file__).resolve().parents[2]
GPQA_PATH = ROOT / "datasets" / "test_data" / "GPQA" / "test.parquet"
SCIBENCH_DIR = ROOT / "datasets" / "test_data" / "SciBench"
REWARD_PATH = ROOT / "scripts" / "reward" / "openscience_reasoning2.py"


def load_reward_module():
    spec = importlib.util.spec_from_file_location("openscience_reward", REWARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reward module: {REWARD_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_gpqa() -> list[dict]:
    table = pq.read_table(GPQA_PATH)
    items = []
    for row in table.to_pylist():
        items.append(
            {
                "benchmark": "GPQA-Diamond",
                "group": "all",
                "id": row["id"],
                "question": row["prompt"][0]["content"],
                "answer": row["reward_model"]["ground_truth"],
                "answer_type": "mcq",
            }
        )
    return items


def load_scibench() -> list[dict]:
    items = []
    for path in sorted(SCIBENCH_DIR.glob("scibench_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            answer = row.get("answer_number")
            if answer is None or str(answer).strip() == "":
                continue
            items.append(
                {
                    "benchmark": "SciBench",
                    "group": row["source"],
                    "id": f"{row['source']}:{row['problemid'].strip()}",
                    "question": (
                        row["problem_text"].strip()
                        + "\n\nPlease reason step by step, and put your final numerical answer within \\boxed{}."
                    ),
                    "answer": str(answer).strip(),
                    "answer_type": "numeric",
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=7168)
    parser.add_argument("--n", type=int, default=1, help="Number of samples per item for avg@n evaluation.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable the model's thinking mode in the official chat template.",
    )
    args = parser.parse_args()

    if not Path(args.model, "config.json").is_file():
        raise FileNotFoundError(f"Missing merged model: {args.model}")
    if not GPQA_PATH.is_file():
        raise FileNotFoundError(f"Missing GPQA-Diamond: {GPQA_PATH}")
    if not SCIBENCH_DIR.is_dir():
        raise FileNotFoundError(f"Missing SciBench directory: {SCIBENCH_DIR}")

    items = load_gpqa() + load_scibench()
    expected_total = 198 + 580
    if len(items) != expected_total:
        raise ValueError(f"Expected {expected_total} evaluation items, found {len(items)}")

    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        for item in items
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ),
        use_tqdm=True,
    )

    reward = load_reward_module()
    records = []
    summary = defaultdict(
        lambda: {
            "count": 0,
            "correct": 0,
            "avg_correct": 0.0,
            "by_group": defaultdict(lambda: {"count": 0, "correct": 0, "avg_correct": 0.0}),
        }
    )
    for item, output in zip(items, outputs, strict=True):
        sample_scores = [
            reward.compute_score(sample.text, item["answer"], item["answer_type"])
            for sample in output.outputs
        ]
        responses = [sample.text for sample in output.outputs]
        correct = int(sample_scores[0]["score"])
        avg_correct = sum(float(score["score"]) for score in sample_scores) / len(sample_scores)
        record = {
            **item,
            "response": responses[0],
            "prediction": sample_scores[0]["pred"],
            "correct": correct,
            "sample_scores": [int(score["score"]) for score in sample_scores],
            "avg_score": avg_correct,
        }
        records.append(record)
        stats = summary[item["benchmark"]]
        stats["count"] += 1
        stats["correct"] += correct
        stats["avg_correct"] += avg_correct
        group_stats = stats["by_group"][item["group"]]
        group_stats["count"] += 1
        group_stats["correct"] += correct
        group_stats["avg_correct"] += avg_correct

    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    final_summary = {}
    for benchmark, stats in summary.items():
        groups = {
            group: {
                **group_stats,
                "accuracy": group_stats["avg_correct"] / group_stats["count"],
                "avg_accuracy": group_stats["avg_correct"] / group_stats["count"],
            }
            for group, group_stats in sorted(stats["by_group"].items())
        }
        final_summary[benchmark] = {
            "count": stats["count"],
            "correct": stats["correct"],
            "avg_correct": stats["avg_correct"],
            "accuracy": stats["avg_correct"] / stats["count"],
            "avg_accuracy": stats["avg_correct"] / stats["count"],
            "by_group": groups,
        }
    metadata = {
        "model": args.model,
        "protocol": {
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "thinking": args.enable_thinking,
            "scoring": "strict MCQ final-letter / numeric final-value rule verifier",
        },
        "benchmarks": final_summary,
    }
    (output_dir / "results.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
