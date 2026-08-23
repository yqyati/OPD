"""One frozen, single-GPU teacher prefill implementation for MOPD services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class PrefillPayload:
    """CPU tensors needed to score one domain-homogeneous student sub-batch."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    responses: torch.Tensor
    student_top_k_ids: torch.Tensor

    @classmethod
    def from_mapping(cls, data: Mapping[str, torch.Tensor]) -> PrefillPayload:
        required = ("input_ids", "attention_mask", "position_ids", "responses", "student_top_k_ids")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"MOPD prefill payload missing fields: {missing}")
        payload = cls(**{name: data[name] for name in required})
        if payload.input_ids.ndim != 2 or payload.responses.ndim != 2 or payload.student_top_k_ids.ndim != 3:
            raise ValueError("MOPD prefill expects input_ids/responses rank 2 and student_top_k_ids rank 3")
        if (
            payload.input_ids.shape[0] != payload.responses.shape[0]
            or payload.responses.shape[:2] != payload.student_top_k_ids.shape[:2]
        ):
            raise ValueError("MOPD prefill batch/response dimensions are inconsistent")
        return payload

    def cpu(self) -> PrefillPayload:
        return PrefillPayload(**{name: getattr(self, name).detach().cpu() for name in self.__dataclass_fields__})


class FrozenTeacherPrefill:
    """Loads one teacher and returns its log-probs on student top-k candidates.

    This uses the same sampled-token payload as existing ``only_stu`` OPD:
    [batch, response_tokens, student_top_k].  It deliberately does not expose a
    teacher prefix loss or a second actor loss; Prefix-MOPD will use the existing
    actor-side prefix fields after its cached-prefix manifest is ready.
    """

    def __init__(self, model_path: str, micro_batch_size: int = 1, dtype: str = "bfloat16") -> None:
        self.model_path = model_path
        self.micro_batch_size = micro_batch_size
        self.dtype = getattr(torch, dtype)
        self.model = None

    def init_model(self) -> None:
        from transformers import AutoModelForCausalLM

        kwargs = {"torch_dtype": self.dtype, "trust_remote_code": True}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, attn_implementation="flash_attention_2", **kwargs
            )
        except Exception as exc:
            # Some checkpoint/model combinations do not expose FlashAttention;
            # the service remains correct with Transformers' normal attention.
            print(f"MOPD teacher {self.model_path}: FlashAttention unavailable ({exc}); using default attention")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        self.model.eval().cuda()

    @torch.inference_mode()
    def score(self, raw_payload: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("FrozenTeacherPrefill.init_model() must run before score()")
        payload = PrefillPayload.from_mapping(raw_payload)
        outputs: list[torch.Tensor] = []
        for start in range(0, payload.input_ids.shape[0], self.micro_batch_size):
            end = min(start + self.micro_batch_size, payload.input_ids.shape[0])
            input_ids = payload.input_ids[start:end].cuda(non_blocking=True)
            attention_mask = payload.attention_mask[start:end].cuda(non_blocking=True)
            position_ids = payload.position_ids[start:end].cuda(non_blocking=True)
            top_k_ids = payload.student_top_k_ids[start:end].cuda(non_blocking=True)
            response_len = top_k_ids.shape[1]
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            # Logit i predicts token i+1.  Responses occupy the right edge of
            # OPD's full input, so this is identical to the existing RM worker's
            # ``[:, -response_length - 1 : -1]`` alignment.
            response_logits = model_output.logits[:, -response_len - 1 : -1, :]
            normalizer = torch.logsumexp(response_logits.float(), dim=-1, keepdim=True)
            candidate_logits = torch.gather(response_logits, dim=-1, index=top_k_ids)
            outputs.append((candidate_logits.float() - normalizer).cpu())
            del model_output, response_logits, normalizer, candidate_logits
        return {"teacher_on_student_log_probs": torch.cat(outputs, dim=0)}
