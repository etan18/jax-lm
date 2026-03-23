import os
from typing import TypeAlias

import numpy as np
import torch


class MemMapDataset:
    """Memory-efficient dataset using np.memmap for large datasets."""

    def __init__(self, data_path: str, verbose: bool = True) -> None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        if verbose:
            print(f"Loaded dataset from {data_path} with {len(self.data):,} tokens")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class DummyDataset:
    """Random token data for benchmarking. Same interface as MemMapDataset."""

    def __init__(self, vocab_size: int, num_tokens: int = 1_000_000, verbose: bool = True) -> None:
        self.data = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.uint16)
        if verbose:
            print(f"Created dummy dataset with {len(self.data):,} random tokens (vocab_size={vocab_size})")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


DatasetLike: TypeAlias = MemMapDataset | DummyDataset


def get_batch_from_memmap(
    dataset: DatasetLike,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a language-modeling batch from a 1D token dataset."""
    max_start = len(dataset.data) - context_length
    if max_start <= 0:
        raise ValueError(f"Dataset too small for context_length={context_length} (len={len(dataset.data)}).")

    start_indices = torch.randint(0, max_start, (batch_size,))
    batch = []
    for start in start_indices.tolist():
        batch_item = torch.tensor(dataset.data[start : start + context_length + 1], dtype=torch.long)
        batch.append(batch_item)
    batch_tensor = torch.stack(batch, dim=0).to(device)
    return batch_tensor[:, :-1], batch_tensor[:, 1:]
