"""SFT dataset that consumes exact precomputed input IDs and token loss masks."""

from __future__ import annotations

import bisect
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
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
        self._files = []
        self._file_starts = []
        total_rows = 0
        for path in parquet_files:
            file = parquet.ParquetFile(copy_local_path_from_hdfs(path, verbose=True))
            columns = set(file.schema_arrow.names)
            required = {"precomputed_input_ids", "precomputed_loss_mask"}
            missing = required.difference(columns)
            if missing:
                raise ValueError(f"Missing precomputed SFT columns: {sorted(missing)}")
            row_group_starts = []
            row_group_total = 0
            for group_index in range(file.num_row_groups):
                row_group_starts.append(row_group_total)
                row_group_total += file.metadata.row_group(group_index).num_rows
            self._file_starts.append(total_rows)
            self._files.append((file, row_group_starts, row_group_total))
            total_rows += row_group_total
        self.length = min(total_rows, max_samples) if max_samples > 0 else total_rows
        self._cached_key = None
        self._cached_rows = None
        required = {"precomputed_input_ids", "precomputed_loss_mask"}
        if self.length == 0:
            raise ValueError("Precomputed SFT dataset is empty")
        print(f"precomputed token dataset len: {self.length}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> dict[str, Any]:
        if item < 0 or item >= self.length:
            raise IndexError(item)
        file_index = bisect.bisect_right(self._file_starts, item) - 1
        file, row_group_starts, _ = self._files[file_index]
        local_item = item - self._file_starts[file_index]
        row_group_index = bisect.bisect_right(row_group_starts, local_item) - 1
        row_in_group = local_item - row_group_starts[row_group_index]
        cache_key = (file_index, row_group_index)
        if self._cached_key != cache_key:
            table = file.read_row_group(
                row_group_index,
                columns=["precomputed_input_ids", "precomputed_loss_mask"],
            )
            self._cached_rows = table.to_pylist()
            self._cached_key = cache_key
        row = self._cached_rows[row_in_group]
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
