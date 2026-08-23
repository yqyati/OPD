"""Deterministic domain-balanced sampler for MOPD mixed batches."""
from __future__ import annotations

import random
from collections import defaultdict

from verl.experimental.dataset.sampler import AbstractSampler


class FixedRatioMOPDSampler(AbstractSampler):
    """Yields complete 20-example blocks: 7 Math, 7 Instruct, 6 Code.

    Set PPO generation batch size to a multiple of 20.  This preserves the paper
    ratio exactly in every batch rather than only in the epoch aggregate.
    """
    def __init__(self, data_source, data_config):
        self.dataset = data_source
        self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))
        if self.batch_size % 20:
            raise ValueError("FixedRatioMOPDSampler requires gen_batch_size divisible by 20")
        labels = data_source.dataframe["mopd_domain"]
        self.indices = defaultdict(list)
        for index, domain in enumerate(labels):
            self.indices[str(domain)].append(index)
        expected = {"math", "instruct", "code"}
        if set(self.indices) != expected or any(not self.indices[d] for d in expected):
            raise ValueError(f"MOPD manifest must contain nonempty {expected}; got {set(self.indices)}")
        self.seed = int(data_config.get("seed", 42) or 42)

    def __iter__(self):
        rng = random.Random(self.seed)
        pools = {d: values.copy() for d, values in self.indices.items()}
        for values in pools.values():
            rng.shuffle(values)
        cursors = defaultdict(int)
        block = ["math"] * 7 + ["instruct"] * 7 + ["code"] * 6
        steps = len(self)
        for _ in range(steps // 20):
            mixed = block.copy()
            rng.shuffle(mixed)
            for domain in mixed:
                if cursors[domain] >= len(pools[domain]):
                    rng.shuffle(pools[domain])
                    cursors[domain] = 0
                yield pools[domain][cursors[domain]]
                cursors[domain] += 1

    def __len__(self):
        return (len(self.dataset) // self.batch_size) * self.batch_size
