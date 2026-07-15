"""SFT dataset that consumes exact precomputed input IDs and token loss masks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset

from verl.utils.fs import copy_local_path_from_hdfs


class PrecomputedTokenSFTDataset(Dataset):
    def __init__(self, parquet_files: str | list[str], tokenizer, config=None, max_samples: int = -1):
        config = config or {}
        self.max_length = int(config.get("max_length", 2048))
        self.truncation = config.get("truncation", "error")
        self.pad_mode = config.get("pad_mode", "right")
        self.tokenizer = tokenizer
        if self.pad_mode != "right":
            raise ValueError("PrecomputedTokenSFTDataset requires data.pad_mode=right")
        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(copy_local_path_from_hdfs(path, verbose=True)) for path in parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        if max_samples > 0 and max_samples < len(self.dataframe):
            self.dataframe = self.dataframe.iloc[:max_samples].reset_index(drop=True)
        required = {"precomputed_input_ids", "precomputed_loss_mask"}
        missing = required.difference(self.dataframe.columns)
        if missing:
            raise ValueError(f"Missing precomputed SFT columns: {sorted(missing)}")
        print(f"precomputed token dataset len: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.dataframe.iloc[item]
        input_ids = [int(value) for value in row["precomputed_input_ids"]]
        loss_mask = [int(value) for value in row["precomputed_loss_mask"]]
        if len(input_ids) != len(loss_mask):
            raise ValueError(f"input/loss-mask length mismatch at row {item}")
        if len(input_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"sequence_length={len(input_ids)} exceeds max_length={self.max_length}")
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            else:
                raise ValueError(f"Unsupported truncation mode: {self.truncation}")
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids.extend([pad_token_id] * pad_len)
        loss_mask.extend([0] * pad_len)
        attention_mask.extend([0] * pad_len)
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "position_ids": torch.arange(self.max_length, dtype=torch.long) * attention_mask_t,
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }
