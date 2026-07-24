"""Parallel binary code verification while preserving token-OPD updates."""

from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import prime_code
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.prime import run_reward_scoring


def binary_code_score(data_source, solution_str, ground_truth, extra_info=None):
    """Return 1 only when a completion passes its complete unit-test suite."""
    if data_source not in {"apps", "codecontests", "codeforces", "taco"}:
        return 0.0
    try:
        score, _ = prime_code.compute_score(solution_str, ground_truth, continuous=False)
        return float(bool(score))
    except Exception:
        return 0.0


@register("parallel_opd_code")
class ParallelOPDCodeRewardManager(AbstractRewardManager):
    """Compute true code reward concurrently but keep ``rm_scores`` as the loss reward."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", num_processes=32, **kwargs):
        self.tokenizer = tokenizer
        self.reward_fn_key = reward_fn_key
        self.num_processes = int(num_processes)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        prompt_length = data.batch["prompts"].shape[-1]
        response_ids = data.batch["responses"]
        response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        completions = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [item.non_tensor_batch["reward_model"]["ground_truth"] for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = run_reward_scoring(
            binary_code_score,
            completions=completions,
            references=ground_truths,
            tasks=data_sources,
            num_processes=self.num_processes,
        )

        true_reward = torch.zeros_like(response_ids, dtype=torch.float32)
        for index, (score, length) in enumerate(zip(scores, response_lengths.tolist(), strict=True)):
            if length > 0:
                true_reward[index, length - 1] = score

        # Standard token OPD optimizes its teacher-derived rm_scores. The code
        # verifier is retained as a true-reward metric only.
        reward_tensor = data.batch.get("rm_scores", true_reward)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {"true_reward_score": true_reward},
            }
        return reward_tensor
