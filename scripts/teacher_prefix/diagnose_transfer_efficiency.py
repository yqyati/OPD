#!/usr/bin/env python3
"""Measure whether a teacher-prefix state transfers from teacher to student.

This is an offline diagnostic. It reuses completed student handoff rollouts,
generates teacher continuations from the same exact prefix states, and reports:

  V_teacher(h) - V_student(h)     (transfer gap)
  V_student(h) / V_teacher(h)     (transfer efficiency)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from diagnose_handoff_value import build_requests, compact_row, worker_process  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_reward(records: list[dict[str, Any]]) -> float:
    return sum(record["rule_score"] for record in records) / len(records)


def summarize_transfer(
    student_records: list[dict[str, Any]],
    teacher_records: list[dict[str, Any]],
    output_dir: Path,
    prefix_lengths: list[int],
) -> None:
    def group(records: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[(int(record["source_row"]), int(record["requested_prefix_len"]))].append(record)
        return grouped

    student_by_state = group(student_records)
    teacher_by_state = group(teacher_records)
    expected_states = set(student_by_state)
    missing = expected_states.difference(teacher_by_state)
    extra = set(teacher_by_state).difference(expected_states)
    if missing or extra:
        raise RuntimeError(
            f"student/teacher state mismatch: missing_teacher={len(missing)}, extra_teacher={len(extra)}"
        )

    per_state = []
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for state in sorted(expected_states):
        student_group = student_by_state[state]
        teacher_group = teacher_by_state[state]
        student_value = mean_reward(student_group)
        teacher_value = mean_reward(teacher_group)
        source_row, prefix_length = state
        row = {
            "source_row": source_row,
            "dataset_index": student_group[0]["dataset_index"],
            "requested_prefix_len": prefix_length,
            "effective_prefix_len": student_group[0]["effective_prefix_len"],
            "student_continuations": len(student_group),
            "teacher_continuations": len(teacher_group),
            "V_student": student_value,
            "V_teacher": teacher_value,
            "transfer_gap": teacher_value - student_value,
            "transfer_efficiency": student_value / teacher_value if teacher_value > 0 else None,
        }
        per_state.append(row)
        by_length[prefix_length].append(row)

    length_summary = []
    for length in prefix_lengths:
        states = by_length[length]
        if not states:
            continue
        teacher_value = sum(state["V_teacher"] for state in states) / len(states)
        student_value = sum(state["V_student"] for state in states) / len(states)
        length_summary.append(
            {
                "requested_prefix_len": length,
                "num_prompts": len(states),
                "V_teacher": teacher_value,
                "V_student": student_value,
                "transfer_gap": teacher_value - student_value,
                "transfer_efficiency": student_value / teacher_value if teacher_value > 0 else None,
                "teacher_better_prompt_count": sum(state["V_teacher"] > state["V_student"] for state in states),
                "student_better_prompt_count": sum(state["V_student"] > state["V_teacher"] for state in states),
                "equal_prompt_count": sum(state["V_student"] == state["V_teacher"] for state in states),
            }
        )

    with open(output_dir / "teacher_handoff_rollouts.jsonl", "w", encoding="utf-8") as handle:
        for record in teacher_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / "per_prompt_transfer_values.jsonl", "w", encoding="utf-8") as handle:
        for record in per_state:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "num_prompts": len({state["source_row"] for state in per_state}),
        "num_student_rollouts_reused": len(student_records),
        "num_teacher_rollouts_generated": len(teacher_records),
        "by_prefix_length": length_summary,
    }
    with open(output_dir / "transfer_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Teacher-prefix parquet used by the student handoff diagnosis.")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-handoff-dir", required=True, help="Completed output directory from diagnose_handoff_value.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--request-batch-size", type=int, default=32)
    args = parser.parse_args()

    student_dir = Path(args.student_handoff_dir)
    student_records_path = student_dir / "handoff_rollouts.jsonl"
    config_path = student_dir / "config.json"
    if not student_records_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("--student-handoff-dir must contain handoff_rollouts.jsonl and config.json")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    student_config = json.loads(config_path.read_text())
    student_records = load_jsonl(student_records_path)
    prefix_lengths = [int(length) for length in student_config["prefix_lengths"]]
    selected_rows = [int(row) for row in student_config["selected_source_rows"]]
    num_continuations = int(student_config["continuations_per_prefix"])
    max_tokens = int(student_config["max_tokens"])
    temperature = float(student_config["temperature"])
    top_p = float(student_config["top_p"])
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU")

    frame = pd.read_parquet(args.input)
    samples = [compact_row(frame.iloc[row].to_dict(), row) for row in selected_rows]
    expected_student_rollouts = len(samples) * len(prefix_lengths) * num_continuations
    if len(student_records) != expected_student_rollouts:
        raise RuntimeError(
            f"student rollout count mismatch: expected={expected_student_rollouts}, got={len(student_records)}"
        )

    config = {
        "input": args.input,
        "teacher_model": args.teacher_model,
        "student_handoff_dir": str(student_dir),
        "prefix_lengths": prefix_lengths,
        "selected_source_rows": selected_rows,
        "continuations_per_prefix": num_continuations,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "gpus": gpu_ids,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    sample_shards = [[] for _ in gpu_ids]
    for index, sample in enumerate(samples):
        sample_shards[index % len(gpu_ids)].append(sample)
    max_model_len = 2048 + max(prefix_lengths) + max_tokens
    worker_args = []
    for worker_id, (gpu_id, shard) in enumerate(zip(gpu_ids, sample_shards)):
        if not shard:
            continue
        worker_args.append(
            (
                worker_id,
                gpu_id,
                args.teacher_model,
                shard,
                prefix_lengths,
                num_continuations,
                max_tokens,
                temperature,
                top_p,
                args.request_batch_size,
                max_model_len,
                str(output_dir / f"teacher_rollouts_worker_{worker_id:02d}.jsonl"),
            )
        )

    print(
        f"reusing {len(student_records)} student rollouts; generating "
        f"{expected_student_rollouts} teacher rollouts on GPUs {gpu_ids}",
        flush=True,
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        worker_outputs = list(executor.map(worker_process, worker_args))
    teacher_records = []
    for path in worker_outputs:
        teacher_records.extend(load_jsonl(Path(path)))
    summarize_transfer(student_records, teacher_records, output_dir, prefix_lengths)


if __name__ == "__main__":
    main()
