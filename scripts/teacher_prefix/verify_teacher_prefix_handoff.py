#!/usr/bin/env python3
"""Verify exact teacher-prefix to student-suffix token handoff alignment."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import _build_teacher_prefix_sft_mask
from verl.utils.model import compute_position_id_with_mask
from verl.utils import torch_functional as verl_F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[0, 64, 128, 256, 512])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--show-row", type=int, default=None)
    parser.add_argument("--show-prefix-length", type=int, default=128)
    parser.add_argument("--preview-chars", type=int, default=1600)
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--student-max-new-tokens", type=int, default=128)
    return parser.parse_args()


def read_rows(path: str, count: int) -> list[dict]:
    parquet = pq.ParquetFile(path)
    rows: list[dict] = []
    for batch in parquet.iter_batches(batch_size=count, columns=["prompt", "teacher_prefix_token_ids"]):
        rows.extend(batch.to_pylist())
        if len(rows) >= count:
            break
    return rows[:count]


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = read_rows(args.dataset, args.samples)
    if not rows:
        raise RuntimeError("dataset contains no rows")

    expected_raw_prompts: list[list[int]] = []
    input_rows: list[torch.Tensor] = []
    prefix_masks: list[torch.Tensor] = []
    selected_lengths: list[int] = []

    for row_idx, row in enumerate(rows):
        messages = row["prompt"]
        base_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prefix_ids = [int(token_id) for token_id in row["teacher_prefix_token_ids"]]
        requested_len = args.prefix_lengths[row_idx % len(args.prefix_lengths)]
        selected_len = min(requested_len, len(prefix_ids))
        full_ids = base_ids + prefix_ids
        if len(full_ids) > args.max_prompt_length:
            raise RuntimeError(f"row {row_idx} exceeds max prompt length: {len(full_ids)}")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=torch.tensor([full_ids], dtype=torch.long),
            attention_mask=torch.ones((1, len(full_ids)), dtype=torch.long),
            max_length=args.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation="error",
        )
        expected_raw_prompts.append(base_ids + prefix_ids[:selected_len])
        input_rows.append(input_ids[0])
        prefix_masks.append(
            _build_teacher_prefix_sft_mask(
                base_prompt_ids=base_ids,
                full_prompt_ids=full_ids,
                max_prompt_length=args.max_prompt_length,
                truncation="error",
            )
        )
        selected_lengths.append(selected_len)

    input_ids = torch.stack(input_rows)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "teacher_prefix_sft_mask": torch.stack(prefix_masks),
            "teacher_prefix_is_complete": torch.zeros(len(rows), dtype=torch.float32),
            "teacher_prefix_base_opd_loss_mask": torch.ones(len(rows), dtype=torch.float32),
            "opd_loss_mask": torch.ones(len(rows), dtype=torch.float32),
        },
        non_tensors={"raw_prompt_ids": np.array([row.tolist() for row in input_ids], dtype=object)},
    )
    selection = DataProto.from_dict(
        tensors={
            "online_prefix_selected_len": torch.tensor(selected_lengths, dtype=torch.long),
            "online_prefix_selected_score": torch.zeros(len(rows)),
            "online_prefix_score_max": torch.zeros(len(rows)),
            "online_prefix_score_mean": torch.zeros(len(rows)),
        }
    )

    # Exercise the production truncation implementation rather than a copy.
    fake_trainer = SimpleNamespace(tokenizer=tokenizer)
    RayPPOTrainer._apply_online_prefix_selection(fake_trainer, batch, selection, metrics={})

    for row_idx, expected_ids in enumerate(expected_raw_prompts):
        actual_raw_ids = [int(token_id) for token_id in batch.non_tensor_batch["raw_prompt_ids"][row_idx]]
        valid_ids = batch.batch["input_ids"][row_idx][batch.batch["attention_mask"][row_idx].bool()].tolist()
        if actual_raw_ids != expected_ids or valid_ids != expected_ids:
            raise AssertionError(f"row {row_idx}: vLLM prompt or valid model input diverged from base+prefix[:L]")

        prefix_logit_positions = torch.nonzero(
            batch.batch["teacher_prefix_sft_mask"][row_idx] > 0, as_tuple=False
        ).flatten()
        selected_len = selected_lengths[row_idx]
        if prefix_logit_positions.numel() != selected_len:
            raise AssertionError(f"row {row_idx}: expected {selected_len} prefix CE positions")
        if selected_len:
            ce_targets = batch.batch["input_ids"][row_idx, prefix_logit_positions + 1].tolist()
            expected_targets = expected_ids[-selected_len:]
            if ce_targets != expected_targets:
                raise AssertionError(f"row {row_idx}: prefix CE targets are off by one token")

    print(
        "PASS: verified exact base + teacher_prefix[:L] handoff for "
        f"{len(rows)} real rows; requested lengths={args.prefix_lengths}."
    )
    if args.show_row is not None:
        if not 0 <= args.show_row < len(rows):
            raise ValueError(f"--show-row must be in [0, {len(rows) - 1}]")
        prefix_ids = [int(token_id) for token_id in rows[args.show_row]["teacher_prefix_token_ids"]]
        handoff = min(args.show_prefix_length, len(prefix_ids))
        prefix_text = tokenizer.decode(prefix_ids[:handoff], skip_special_tokens=False)
        tail_text = tokenizer.decode(prefix_ids[handoff:], skip_special_tokens=False)
        print(f"\n===== Teacher prefix, first {handoff} tokens =====")
        print(prefix_text[-args.preview_chars:])
        print("\n===== Stored teacher continuation, immediately after handoff =====")
        print(tail_text[: args.preview_chars])
        if args.student_model is not None:
            from transformers import AutoModelForCausalLM

            prompt_ids = tokenizer.apply_chat_template(
                rows[args.show_row]["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            ) + prefix_ids[:handoff]
            print("\n===== Student continuation from the exact same token prompt =====")
            model = AutoModelForCausalLM.from_pretrained(
                args.student_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to("cpu")
            model.eval()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=torch.tensor([prompt_ids], dtype=torch.long),
                    attention_mask=torch.ones((1, len(prompt_ids)), dtype=torch.long),
                    max_new_tokens=args.student_max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            student_ids = generated[0, len(prompt_ids) :].tolist()
            print(tokenizer.decode(student_ids, skip_special_tokens=False)[: args.preview_chars])


if __name__ == "__main__":
    main()
