#!/usr/bin/env python3
"""Test whether interaction-mass summaries predict teacher-prefix handoff value.

The handoff values come from completed student rollouts.  Scores are computed
only from the exact teacher-prefix token IDs and frozen student/teacher forward
distributions, with no access to continuation responses or rewards.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


TOP_K = 16


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_messages(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return [dict(item) for item in prompt]


def worker(args: tuple[Any, ...]) -> str:
    worker_id, gpu_id, input_path, student_model, teacher_model, samples, lengths, batch_size, output_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frame = pd.read_parquet(input_path)
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = []
    max_length = max(lengths)
    for item in samples:
        row = frame.iloc[item["source_row"]].to_dict()
        base_ids = tokenizer.apply_chat_template(
            prompt_messages(row["prompt"]), tokenize=True, add_generation_prompt=True, enable_thinking=True
        )
        prefix_ids = [int(token) for token in row["teacher_prefix_token_ids"]][:max_length]
        if len(prefix_ids) < max_length:
            raise RuntimeError(f"source_row={item['source_row']} lacks exact prefix length {max_length}")
        prepared.append({**item, "base_ids": base_ids, "prefix_ids": prefix_ids})

    student = teacher = None
    try:
        common = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
        student = AutoModelForCausalLM.from_pretrained(student_model, **common).eval().cuda()
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model, **common).eval().cuda()
        with open(output_path, "w", encoding="utf-8") as handle:
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                sequences = [item["base_ids"] + item["prefix_ids"] for item in batch]
                padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
                input_ids = padded["input_ids"].cuda()
                attention_mask = padded["attention_mask"].cuda()
                starts = [len(item["base_ids"]) for item in batch]
                ends = [len(sequence) for sequence in sequences]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    s_logits = student(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                    t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                for row_idx, item in enumerate(batch):
                    s = s_logits[row_idx, starts[row_idx] - 1 : ends[row_idx] - 1].float()
                    t = t_logits[row_idx, starts[row_idx] - 1 : ends[row_idx] - 1].float()
                    teacher_prefix_ids = input_ids[row_idx, starts[row_idx] : ends[row_idx]]
                    s_logp = s - torch.logsumexp(s, dim=-1, keepdim=True)
                    t_logp = t - torch.logsumexp(t, dim=-1, keepdim=True)
                    s_actual_logp = s_logp.gather(-1, teacher_prefix_ids.unsqueeze(-1)).squeeze(-1)
                    t_actual_logp = t_logp.gather(-1, teacher_prefix_ids.unsqueeze(-1)).squeeze(-1)
                    s_top_logits, s_ids = torch.topk(s, k=TOP_K, dim=-1)
                    t_top_logits, t_ids = torch.topk(t, k=TOP_K, dim=-1)
                    s_probs = (s_top_logits - torch.logsumexp(s, dim=-1, keepdim=True)).exp()
                    t_probs = (t_top_logits - torch.logsumexp(t, dim=-1, keepdim=True)).exp()
                    eq = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2))
                    # Set overlap deliberately ignores probability mass.  Keep it
                    # as a separate negative-control signal from interaction mass.
                    overlap = eq.any(dim=-1).float().mean(dim=-1)
                    interaction = torch.minimum(
                        (s_probs * eq.any(dim=-1)).sum(dim=-1),
                        (t_probs * eq.any(dim=-2)).sum(dim=-1),
                    )
                    # All quantities below use only the observed teacher prefix.
                    # This is the teacher top-k approximation to forward KL at each
                    # prefix position, not a score from a student continuation.
                    student_at_teacher_top = s_logp.gather(-1, t_ids)
                    teacher_prefix_forward_kl = (
                        t_probs * (t_top_logits - torch.logsumexp(t, dim=-1, keepdim=True) - student_at_teacher_top)
                    ).sum(dim=-1)
                    for length in lengths:
                        curve = interaction[:length]
                        handle.write(
                            json.dumps(
                                {
                                    "source_row": item["source_row"],
                                    "prefix_len": length,
                                    "V_student": item["values"][length],
                                    "median_interaction_mass": float(torch.median(curve)),
                                    "mean_interaction_mass": float(curve.mean()),
                                    "final_interaction_mass": float(curve[-1]),
                                    "median_top16_overlap_ratio": float(torch.median(overlap[:length])),
                                    "mean_top16_overlap_ratio": float(overlap[:length].mean()),
                                    "final_top16_overlap_ratio": float(overlap[length - 1]),
                                    # Teacher-prefix reachability: can the student
                                    # locally predict the exact teacher scaffold?
                                    "student_teacher_token_support": float(s_actual_logp[:length].mean()),
                                    "teacher_teacher_token_support": float(t_actual_logp[:length].mean()),
                                    "student_minus_teacher_advantage": float(
                                        (s_actual_logp[:length] - t_actual_logp[:length]).mean()
                                    ),
                                    "teacher_minus_student_advantage": float(
                                        (t_actual_logp[:length] - s_actual_logp[:length]).mean()
                                    ),
                                    "teacher_prefix_top16_forward_kl": float(teacher_prefix_forward_kl[:length].mean()),
                                }
                            )
                            + "\n"
                        )
                del s_logits, t_logits
                print(f"[worker {worker_id}, gpu {gpu_id}] {min(start + len(batch), len(prepared))}/{len(prepared)} prompts", flush=True)
    finally:
        del student, teacher
        gc.collect()
        torch.cuda.empty_cache()
    return output_path


def descending_percentile_ranks(scores: list[float]) -> list[float]:
    """Return average percentile ranks, where larger scores have larger ranks."""
    count = len(scores)
    ranks = []
    for score in scores:
        lower = sum(other < score for other in scores)
        equal = sum(other == score for other in scores)
        ranks.append((lower + (equal - 1) / 2) / max(count - 1, 1))
    return ranks


def summarize_selector(
    by_prompt: dict[int, list[dict[str, Any]]],
    lengths: list[int],
    selector_name: str,
    score_fn: Any,
) -> dict[str, Any]:
    selected_values, random_values, oracle_values, pairwise = [], [], [], []
    selected_lengths = defaultdict(int)
    for group in by_prompt.values():
        group = sorted(group, key=lambda item: item["prefix_len"])
        values = [item["V_student"] for item in group]
        scores = score_fn(group)
        top = max(scores)
        # The online policy must make one decision.  Resolve exact score ties by
        # taking the earliest handoff, rather than injecting another hyperparameter.
        selected_index = next(index for index, score in enumerate(scores) if score == top)
        selected_values.append(values[selected_index])
        selected_lengths[group[selected_index]["prefix_len"]] += 1
        random_values.append(sum(values) / len(values))
        oracle_values.append(max(values))
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                if values[left] == values[right]:
                    continue
                direction = (scores[left] > scores[right]) - (scores[left] < scores[right])
                truth = (values[left] > values[right]) - (values[left] < values[right])
                pairwise.append(1.0 if direction == truth else 0.5 if direction == 0 else 0.0)
    return {
        "selected_V_student": sum(selected_values) / len(selected_values),
        "random_V_student": sum(random_values) / len(random_values),
        "uplift_over_random": (sum(selected_values) - sum(random_values)) / len(selected_values),
        "oracle_V_student": sum(oracle_values) / len(oracle_values),
        "pairwise_accuracy_on_unequal_value_pairs": sum(pairwise) / len(pairwise) if pairwise else None,
        "selected_length_counts_earliest_tie_break": {str(length): selected_lengths[length] for length in lengths},
    }


def summarize(rows: list[dict[str, Any]], output_dir: Path, lengths: list[int]) -> None:
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["source_row"]].append(row)
    metrics = [
        "median_interaction_mass",
        "mean_interaction_mass",
        "final_interaction_mass",
        "median_top16_overlap_ratio",
        "mean_top16_overlap_ratio",
        "final_top16_overlap_ratio",
        "student_teacher_token_support",
        "teacher_teacher_token_support",
        "student_minus_teacher_advantage",
        "teacher_minus_student_advantage",
        "teacher_prefix_top16_forward_kl",
    ]
    summary = {}
    for metric in metrics:
        summary[metric] = summarize_selector(by_prompt, lengths, metric, lambda group, name=metric: [item[name] for item in group])

    # No thresholds or tunable weights: each component gets an within-trajectory
    # percentile rank and the selector maximizes either their sum or bottleneck.
    for mass_name in ("mean_interaction_mass", "median_interaction_mass"):
        for opportunity_name in ("teacher_prefix_top16_forward_kl", "teacher_minus_student_advantage"):
            for aggregation in ("rank_sum", "rank_bottleneck"):
                name = f"{aggregation}_{mass_name}_and_{opportunity_name}"

                def combined_score(
                    group: list[dict[str, Any]],
                    mass_name: str = mass_name,
                    opportunity_name: str = opportunity_name,
                    aggregation: str = aggregation,
                ) -> list[float]:
                    mass_ranks = descending_percentile_ranks([item[mass_name] for item in group])
                    opportunity_ranks = descending_percentile_ranks([item[opportunity_name] for item in group])
                    if aggregation == "rank_sum":
                        return [left + right for left, right in zip(mass_ranks, opportunity_ranks)]
                    return [min(left, right) for left, right in zip(mass_ranks, opportunity_ranks)]

                summary[name] = summarize_selector(by_prompt, lengths, name, combined_score)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(output_dir / "interaction_handoff_scores.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--handoff-values", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    handoff = read_jsonl(Path(args.handoff_values))
    lengths = sorted(int(length) for length in handoff[0]["continuation_value_by_prefix_len"])
    samples = [
        {
            "source_row": item["source_row"],
            "values": {int(length): float(value) for length, value in item["continuation_value_by_prefix_len"].items()},
        }
        for item in handoff
    ]
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    shards = [[] for _ in gpu_ids]
    for index, sample in enumerate(samples):
        shards[index % len(shards)].append(sample)
    context = multiprocessing.get_context("spawn")
    worker_args = [
        (index, gpu, args.input, args.student_model, args.teacher_model, shard, lengths, args.batch_size, str(output_dir / f"scores_worker_{index:02d}.jsonl"))
        for index, (gpu, shard) in enumerate(zip(gpu_ids, shards))
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        paths = list(executor.map(worker, worker_args))
    summarize([row for path in paths for row in read_jsonl(Path(path))], output_dir, lengths)


if __name__ == "__main__":
    main()
