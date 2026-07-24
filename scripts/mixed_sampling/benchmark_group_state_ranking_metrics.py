#!/usr/bin/env python3
"""Benchmark forward-only policy-interaction scores for Group OPD states.

Each score sees only a question and one already-generated student prefix.  The
stored continuation value V_student is used only afterwards as a label for
within-question ranking quality; it is never part of score computation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


TOP_K = 16


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def attach_prefix_token_ids(state_values: list[dict[str, Any]], continuation_path: Path) -> list[dict[str, Any]]:
    """Recover generated prefix ids, which are intentionally omitted from state-value summaries."""
    prefix_ids: dict[tuple[int, int], list[int]] = {}
    for record in read_jsonl(continuation_path):
        key = (record["source_row"], record["prefix_id"])
        prefix_ids.setdefault(key, record["prefix_token_ids"])
    missing = []
    enriched = []
    for state in state_values:
        key = (state["source_row"], state["prefix_id"])
        ids = prefix_ids.get(key)
        if ids is None:
            missing.append(key)
            continue
        enriched.append({**state, "prefix_token_ids": ids})
    if missing:
        raise RuntimeError(f"missing generated prefix token ids for {len(missing)} states; first={missing[:3]}")
    return enriched


def load_prompt_ids(input_path: str, states: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    frame = pd.read_parquet(input_path)
    results = []
    prompt_cache: dict[int, list[int]] = {}
    for state in states:
        source_row = state["source_row"]
        if source_row not in prompt_cache:
            prompt = frame.iloc[source_row].to_dict()["prompt"]
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            prompt_cache[source_row] = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, enable_thinking=True
            )
        results.append({**state, "prompt_token_ids": prompt_cache[source_row]})
    return results


def policy_features(logits: Any, input_ids: Any, starts: list[int], ends: list[int]) -> dict[str, list[float]]:
    """Compute the same top-k interaction quantities used by online selection."""
    import torch

    metrics: dict[str, list[float]] = defaultdict(list)
    for row, (start, end) in enumerate(zip(starts, ends)):
        # logits[position] predicts input_ids[position + 1].
        token_logits = logits[row, start - 1 : end - 1].float()
        target_ids = input_ids[row, start:end]
        log_probs = token_logits - torch.logsumexp(token_logits, dim=-1, keepdim=True)
        metrics["token_logprob"].append(float(log_probs.gather(-1, target_ids.unsqueeze(-1)).mean()))
        # Retained only for the 128 prefix positions. This lets us score soft
        # cross-policy coverage without imposing a hard top-k intersection.
        metrics["full_log_probs"].append(log_probs)

        top_logits, top_ids = torch.topk(token_logits, k=TOP_K, dim=-1)
        top_log_probs = top_logits - torch.logsumexp(token_logits, dim=-1, keepdim=True)
        metrics["top_ids"].append(top_ids)
        metrics["top_probs"].append(top_log_probs.exp())
        top_norm_probs = torch.softmax(top_logits, dim=-1)
        entropy = -(top_norm_probs * top_norm_probs.clamp_min(1e-12).log()).sum(dim=-1)
        metrics["top_entropy"].append(entropy)

        handoff_logits = logits[row, end - 1].float()
        handoff_top_logits, handoff_top_ids = torch.topk(handoff_logits, k=TOP_K)
        handoff_probs = torch.softmax(handoff_top_logits, dim=-1)
        handoff_entropy = -(handoff_probs * handoff_probs.clamp_min(1e-12).log()).sum()
        metrics["handoff_top_ids"].append(handoff_top_ids)
        metrics["handoff_top_probs"].append(handoff_probs)
        metrics["handoff_entropy"].append(handoff_entropy)
    return metrics


def shared_features(student: dict[str, list[float]], teacher: dict[str, list[float]]) -> list[dict[str, float]]:
    import torch

    rows = []
    for index in range(len(student["token_logprob"])):
        s_ids, s_probs = student["top_ids"][index], student["top_probs"][index]
        t_ids, t_probs = teacher["top_ids"][index], teacher["top_probs"][index]
        eq = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2))
        s_shared = eq.any(dim=-1)
        t_shared = eq.any(dim=-2)
        overlap = s_shared.float().mean(dim=-1)
        s_mass = (s_probs * s_shared).sum(dim=-1)
        t_mass = (t_probs * t_shared).sum(dim=-1)
        interaction = torch.minimum(s_mass, t_mass)
        entropy_agreement = torch.exp(-(student["top_entropy"][index] - teacher["top_entropy"][index]).abs())

        # Soft coverage: one policy's top-k actions may be strongly supported
        # by the other policy even if a token lies just outside its own top-k.
        s_log_probs, t_log_probs = student["full_log_probs"][index], teacher["full_log_probs"][index]
        teacher_on_student_topk = t_log_probs.gather(-1, s_ids).exp().sum(dim=-1)
        student_on_teacher_topk = s_log_probs.gather(-1, t_ids).exp().sum(dim=-1)
        soft_shared_min = torch.minimum(teacher_on_student_topk, student_on_teacher_topk)
        soft_shared_geomean = torch.sqrt(teacher_on_student_topk * student_on_teacher_topk)

        # Bhattacharyya coefficient on the exact union of both top-k supports.
        # Count shared ids once, then append teacher-only ids.
        t_only = ~t_ids.unsqueeze(-1).eq(s_ids.unsqueeze(-2)).any(dim=-1)
        bc_s = torch.sqrt(s_probs * t_log_probs.gather(-1, s_ids).exp()).sum(dim=-1)
        t_only_ids = t_ids.masked_fill(~t_only, 0)
        bc_t = torch.sqrt(s_log_probs.gather(-1, t_only_ids).exp() * t_probs) * t_only
        union_bhattacharyya = bc_s + bc_t.sum(dim=-1)

        sh_ids, sh_probs = student["handoff_top_ids"][index], student["handoff_top_probs"][index]
        th_ids, th_probs = teacher["handoff_top_ids"][index], teacher["handoff_top_probs"][index]
        handoff_eq = sh_ids.unsqueeze(-1).eq(th_ids.unsqueeze(-2))
        handoff_s_shared = handoff_eq.any(dim=-1)
        handoff_t_shared = handoff_eq.any(dim=-2)
        handoff_overlap = float(handoff_s_shared.float().mean())
        handoff_interaction = float(
            torch.minimum((sh_probs * handoff_s_shared).sum(), (th_probs * handoff_t_shared).sum())
        )
        handoff_entropy_agreement = float(
            torch.exp(-(student["handoff_entropy"][index] - teacher["handoff_entropy"][index]).abs())
        )
        midpoint = interaction.numel() // 2
        rows.append(
            {
                "teacher_token_support": teacher["token_logprob"][index],
                "student_token_support": student["token_logprob"][index],
                "trajectory_overlap": float(overlap.mean()),
                "trajectory_interaction_mass": float(interaction.mean()),
                "trajectory_overlap_entropy": float((overlap * entropy_agreement).mean()),
                "trajectory_interaction_tail_quarter": float(interaction[-max(1, interaction.numel() // 4) :].mean()),
                "trajectory_interaction_min": float(interaction.min()),
                "trajectory_interaction_q125": float(torch.quantile(interaction, 0.125)),
                "trajectory_interaction_lower_quartile": float(torch.quantile(interaction, 0.25)),
                "trajectory_interaction_q375": float(torch.quantile(interaction, 0.375)),
                "trajectory_interaction_median": float(torch.quantile(interaction, 0.5)),
                "trajectory_interaction_lowest_quarter_mean": float(
                    interaction.sort().values[: max(1, interaction.numel() // 4)].mean()
                ),
                "trajectory_interaction_harmonic_mean": float(
                    interaction.numel() / interaction.clamp_min(1e-8).reciprocal().sum()
                ),
                "trajectory_interaction_mean_minus_std": float(interaction.mean() - interaction.std(unbiased=False)),
                "teacher_on_student_topk_mass": float(teacher_on_student_topk.mean()),
                "student_on_teacher_topk_mass": float(student_on_teacher_topk.mean()),
                "soft_shared_topk_min_mass": float(soft_shared_min.mean()),
                "soft_shared_topk_geomean_mass": float(soft_shared_geomean.mean()),
                "union_topk_bhattacharyya": float(union_bhattacharyya.mean()),
                "late_minus_early_interaction": float(interaction[midpoint:].mean() - interaction[:midpoint].mean()),
                "late_minus_early_overlap": float(overlap[midpoint:].mean() - overlap[:midpoint].mean()),
                "handoff_overlap": handoff_overlap,
                "handoff_interaction_mass": handoff_interaction,
                "handoff_entropy_agreement": handoff_entropy_agreement,
            }
        )
    return rows


def worker_process(args: tuple[Any, ...]) -> str:
    worker_id, gpu_id, student_model, teacher_model, samples, batch_size, output_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = load_prompt_ids(samples["input_path"], samples["states"], tokenizer)
    student = teacher = None
    try:
        common = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
        student = AutoModelForCausalLM.from_pretrained(student_model, **common).eval().cuda()
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model, **common).eval().cuda()
        with open(output_path, "w", encoding="utf-8") as handle:
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                sequences = [sample["prompt_token_ids"] + sample["prefix_token_ids"] for sample in batch]
                padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
                input_ids = padded["input_ids"].cuda()
                attention_mask = padded["attention_mask"].cuda()
                starts = [len(sample["prompt_token_ids"]) for sample in batch]
                ends = [len(sequence) for sequence in sequences]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    student_logits = student(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                    student_features = policy_features(student_logits, input_ids, starts, ends)
                    del student_logits
                    teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                    teacher_features = policy_features(teacher_logits, input_ids, starts, ends)
                    del teacher_logits
                features = shared_features(student_features, teacher_features)
                for sample, feature in zip(batch, features):
                    handle.write(json.dumps({"source_row": sample["source_row"], "prefix_id": sample["prefix_id"], **feature}) + "\n")
                print(f"[worker {worker_id}, gpu {gpu_id}] {min(start + len(batch), len(samples))}/{len(samples)} states", flush=True)
    finally:
        del student, teacher
        gc.collect()
        torch.cuda.empty_cache()
    return output_path


def average_rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        next_cursor = cursor + 1
        while next_cursor < len(order) and values[order[next_cursor]] == values[order[cursor]]:
            next_cursor += 1
        rank = (cursor + next_cursor - 1) / 2 + 1
        for position in range(cursor, next_cursor):
            ranks[order[position]] = rank
        cursor = next_cursor
    return ranks


def benchmark(scores: list[dict[str, Any]], state_values: list[dict[str, Any]]) -> dict[str, Any]:
    value_map = {(row["source_row"], row["prefix_id"]): row["V_student"] for row in state_values}
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        row["V_student"] = value_map[(row["source_row"], row["prefix_id"])]
        by_prompt[row["source_row"]].append(row)
    metric_names = [key for key in scores[0] if key not in {"source_row", "prefix_id", "V_student"}]
    results: dict[str, Any] = {}
    for metric in metric_names:
        selected_values: list[float] = []
        random_values: list[float] = []
        oracle_values: list[float] = []
        pairwise_scores: list[float] = []
        top1_hits: list[float] = []
        nonzero_selected: list[float] = []
        nonzero_random: list[float] = []
        for rows in by_prompt.values():
            values = [row["V_student"] for row in rows]
            raw_scores = [row[metric] for row in rows]
            top_score = max(raw_scores)
            selected = [value for score, value in zip(raw_scores, values) if score == top_score]
            selected_value = sum(selected) / len(selected)
            selected_values.append(selected_value)
            random_values.append(sum(values) / len(values))
            oracle_values.append(max(values))
            target_best = {index for index, value in enumerate(values) if value == max(values)}
            selected_indices = {index for index, score in enumerate(raw_scores) if score == top_score}
            top1_hits.append(len(target_best & selected_indices) / len(target_best | selected_indices))
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    if values[left] == values[right]:
                        continue
                    direction = (raw_scores[left] > raw_scores[right]) - (raw_scores[left] < raw_scores[right])
                    truth = (values[left] > values[right]) - (values[left] < values[right])
                    pairwise_scores.append(1.0 if direction == truth else 0.5 if direction == 0 else 0.0)
            if max(values) > min(values):
                nonzero_selected.append(selected_value)
                nonzero_random.append(sum(values) / len(values))
        results[metric] = {
            "num_prompts": len(by_prompt),
            "selected_V_student": sum(selected_values) / len(selected_values),
            "random_V_student": sum(random_values) / len(random_values),
            "uplift_over_random": (sum(selected_values) - sum(random_values)) / len(selected_values),
            "oracle_V_student": sum(oracle_values) / len(oracle_values),
            "oracle_regret": (sum(oracle_values) - sum(selected_values)) / len(selected_values),
            "pairwise_accuracy_on_unequal_value_pairs": sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else None,
            "top1_jaccard": sum(top1_hits) / len(top1_hits),
            "selected_V_on_nonzero_gap_prompts": sum(nonzero_selected) / len(nonzero_selected),
            "random_V_on_nonzero_gap_prompts": sum(nonzero_random) / len(nonzero_random),
            "nonzero_gap_uplift": (sum(nonzero_selected) - sum(nonzero_random)) / len(nonzero_selected),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--state-values", required=True)
    parser.add_argument(
        "--continuations",
        help="Raw continuation records containing generated prefix_token_ids. Defaults to the state-value sibling file.",
    )
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    state_value_path = Path(args.state_values)
    continuation_path = Path(args.continuations) if args.continuations else state_value_path.with_name("group_prefix_continuations.jsonl")
    if not continuation_path.is_file():
        raise FileNotFoundError(f"continuation records not found: {continuation_path}")
    state_values = attach_prefix_token_ids(read_jsonl(state_value_path), continuation_path)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    random.Random(42).shuffle(state_values)
    shards = [[] for _ in gpu_ids]
    for index, state in enumerate(state_values):
        shards[index % len(shards)].append(state)
    worker_args = [
        (index, gpu, args.student_model, args.teacher_model, {"input_path": args.input, "states": shard}, args.batch_size, str(output_dir / f"scores_worker_{index:02d}.jsonl"))
        for index, (gpu, shard) in enumerate(zip(gpu_ids, shards))
    ]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
        paths = list(executor.map(worker_process, worker_args))
    scores = [record for path in paths for record in read_jsonl(Path(path))]
    results = {
        "all": benchmark(scores, state_values),
        "source_row_even": benchmark([row for row in scores if row["source_row"] % 2 == 0], state_values),
        "source_row_odd": benchmark([row for row in scores if row["source_row"] % 2 == 1], state_values),
    }
    with open(output_dir / "state_ranking_scores.jsonl", "w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
