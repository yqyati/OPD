"""Binary execution reward for standalone code GRPO."""

from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import prime_code
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.prime import run_reward_scoring


def binary_code_score(data_source, solution_str, ground_truth, extra_info=None):
    """Return one iff a completion passes its whole local test suite."""
    if data_source not in {"apps", "codecontests", "codeforces", "taco"}:
        return 0.0
    try:
        score, _ = prime_code.compute_score(solution_str, ground_truth, continuous=False)
        return float(bool(score))
    except Exception:
        return 0.0


@register("parallel_grpo_code")
class ParallelGRPOCodeRewardManager(AbstractRewardManager):
    """Use only concurrent binary execution reward for code GRPO.

    This manager deliberately never reads ``rm_scores``. It therefore cannot
    inherit token-OPD or teacher-derived reward terms.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", num_processes=64, **kwargs):
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

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        for index, (score, length) in enumerate(zip(scores, response_lengths.tolist(), strict=True)):
            if length > 0:
                reward_tensor[index, length - 1] = score

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {"true_reward_score": reward_tensor},
            }
        return reward_tensor
