#!/usr/bin/env python3
"""Analyze how teacher prefixes causally change student answer distributions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def normalized_answer(response: str) -> str | None:
    start = response.rfind("\\boxed{")
    if start < 0:
        return None
    cursor = start + len("\\boxed{")
    depth = 1
    for index in range(cursor, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return (
                    response[cursor:index]
                    .strip()
                    .replace(" ", "")
                    .replace("\n", "")
                    .replace("\\left", "")
                    .replace("\\right", "")
                    .replace("\\dfrac", "\\frac")
                    .replace("\\tfrac", "\\frac")
                    or None
                )
    return None


def read_records(directory: Path) -> tuple[dict[int, dict[int, list[dict[str, Any]]]], int]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    malformed = 0
    for path in sorted(directory.glob("handoff_rollouts_worker_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            grouped[int(row["source_row"])][int(row["requested_prefix_len"])].append(row)
    for by_length in grouped.values():
        for records in by_length.values():
            records.sort(key=lambda row: int(row["continuation_id"]))
    return grouped, malformed


def answer_distribution(records: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    # Distinct invalid placeholders prevent repeated missing boxes from being
    # mistaken for meaningful answer consensus.
    answers = []
    for index, record in enumerate(records):
        answer = normalized_answer(str(record["response"]))
        answers.append(answer if answer is not None else f"__invalid_{index}")
    counts = Counter(answers)
    total = len(records)
    distribution = {answer: count / total for answer, count in counts.items()}
    valid_counts = Counter(answer for answer in answers if not answer.startswith("__invalid_"))
    consensus = (max(valid_counts.values()) / total) if valid_counts else 0.0
    return distribution, consensus


def entropy(distribution: dict[str, float]) -> float:
    return -sum(probability * math.log(probability) for probability in distribution.values() if probability > 0)


def js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left).union(right)
    midpoint = {key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2.0 for key in keys}

    def kl(source: dict[str, float]) -> float:
        return sum(probability * math.log(probability / midpoint[key]) for key, probability in source.items() if probability > 0)

    return (kl(left) + kl(right)) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-rollout-dir", required=True)
    parser.add_argument("--prefix-rollout-dir", required=True)
    parser.add_argument("--prefix-lengths", default="32,64,128,256,512")
    parser.add_argument("--continuations", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lengths = [int(value) for value in args.prefix_lengths.split(",") if value.strip()]
    zero_records, malformed_zero = read_records(Path(args.zero_rollout_dir))
    prefix_records, malformed_prefix = read_records(Path(args.prefix_rollout_dir))
    rows = []
    for source_row in sorted(set(zero_records).intersection(prefix_records)):
        zero = zero_records[source_row].get(0, [])
        if len(zero) != args.continuations or any(
            len(prefix_records[source_row].get(length, [])) != args.continuations for length in lengths
        ):
            continue
        zero_distribution, zero_consensus = answer_distribution(zero)
        zero_entropy = entropy(zero_distribution)
        for length in lengths:
            records = prefix_records[source_row][length]
            distribution, consensus = answer_distribution(records)
            rows.append(
                {
                    "source_row": source_row,
                    "prefix_len": length,
                    "consensus_uplift": consensus - zero_consensus,
                    "entropy_reduction": zero_entropy - entropy(distribution),
                    "answer_distribution_js_shift": js_divergence(zero_distribution, distribution),
                    "mean_rule_reward": sum(float(record["rule_score"]) for record in records) / len(records),
                }
            )

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[int(row["prefix_len"])].append(row)
        by_prompt[int(row["source_row"])].append(row)
    selectors = {}
    for metric in ("consensus_uplift", "entropy_reduction", "answer_distribution_js_shift"):
        counts = Counter()
        selected_rewards = []
        random_rewards = []
        oracle_rewards = []
        pairwise = []
        for prompt_rows in by_prompt.values():
            prompt_rows.sort(key=lambda row: int(row["prefix_len"]))
            highest = max(row[metric] for row in prompt_rows)
            chosen = next(row for row in prompt_rows if row[metric] == highest)
            counts[int(chosen["prefix_len"])] += 1
            selected_rewards.append(float(chosen["mean_rule_reward"]))
            values = [float(row["mean_rule_reward"]) for row in prompt_rows]
            scores = [float(row[metric]) for row in prompt_rows]
            random_rewards.append(sum(values) / len(values))
            oracle_rewards.append(max(values))
            for left in range(len(prompt_rows)):
                for right in range(left + 1, len(prompt_rows)):
                    if values[left] == values[right]:
                        continue
                    direction = (scores[left] > scores[right]) - (scores[left] < scores[right])
                    truth = (values[left] > values[right]) - (values[left] < values[right])
                    pairwise.append(1.0 if direction == truth else 0.5 if direction == 0 else 0.0)
        selectors[metric] = {
            "argmax_length_counts_earliest_tie_break": {str(length): counts[length] for length in lengths},
            "mean_argmax_prefix_len": sum(length * counts[length] for length in lengths) / len(by_prompt),
            "selected_mean_rule_reward": sum(selected_rewards) / len(selected_rewards),
            "random_mean_rule_reward": sum(random_rewards) / len(random_rewards),
            "oracle_mean_rule_reward": sum(oracle_rewards) / len(oracle_rewards),
            "pairwise_accuracy_on_unequal_reward_pairs": sum(pairwise) / len(pairwise) if pairwise else None,
        }
    summary = {
        "complete_matched_prompts": len(by_prompt),
        "malformed_zero_rollout_records_skipped": malformed_zero,
        "malformed_prefix_rollout_records_skipped": malformed_prefix,
        "by_prefix_length": {
            str(length): {
                "mean_consensus_uplift": sum(row["consensus_uplift"] for row in by_length[length]) / len(by_length[length]),
                "mean_entropy_reduction": sum(row["entropy_reduction"] for row in by_length[length]) / len(by_length[length]),
                "mean_answer_distribution_js_shift": sum(row["answer_distribution_js_shift"] for row in by_length[length]) / len(by_length[length]),
                "mean_rule_reward": sum(row["mean_rule_reward"] for row in by_length[length]) / len(by_length[length]),
            }
            for length in lengths
        },
        "argmax_selector_behavior": selectors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
