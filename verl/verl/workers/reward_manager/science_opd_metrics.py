"""Cheap science true-reward logging while preserving standard token-OPD loss."""

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("science_opd_metrics")
class ScienceOPDMetricsRewardManager(AbstractRewardManager):
    """Compute exact-answer metrics, but optimize teacher-derived ``rm_scores``.

    This is intentionally equivalent to the established code OPD behavior: the
    verifier is an observation metric only and never alters the loss reward.
    """

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **_kwargs: Any):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        prompt_length = data.batch["prompts"].shape[-1]
        response_ids = data.batch["responses"]
        response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        true_reward = torch.zeros_like(response_ids, dtype=torch.float32)
        extra = defaultdict(list)

        for index, item in enumerate(data):
            length = int(response_lengths[index].item())
            response = self.tokenizer.decode(response_ids[index, :length], skip_special_tokens=True)
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]
            extra_info = dict(item.non_tensor_batch.get("extra_info", {}))
            result = self.compute_score(
                data_source=data_source,
                solution_str=response,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if not isinstance(result, dict):
                result = {"score": float(result)}
            score = float(result["score"])
            if length > 0:
                true_reward[index, length - 1] = score
            for key, value in result.items():
                extra[key].append(value)

        # Token OPD owns the training loss. ``true_reward_score`` is retained
        # for logs and rollout dumps only.
        reward_tensor = data.batch.get("rm_scores", true_reward)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {"true_reward_score": true_reward, **dict(extra)},
            }
        return reward_tensor
