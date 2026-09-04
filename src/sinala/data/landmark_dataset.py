from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sinala.data.landmark_augmenter import Augmentation


class LandmarkSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        sequence_paths: list[Path],
        labels: list[int],
        expected_sequence_length: int = 32,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if len(sequence_paths) != len(labels):
            raise ValueError("sequence_paths and labels must have the same length")
        if not sequence_paths:
            raise ValueError("Landmark dataset cannot be empty")
        if expected_sequence_length < 2:
            raise ValueError("expected_sequence_length must be at least 2")
        self.sequence_paths = sequence_paths
        self.labels = labels
        self.expected_sequence_length = expected_sequence_length
        self.transform: Augmentation | None = transform

    def __len__(self) -> int:
        return len(self.sequence_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = np.load(self.sequence_paths[index], allow_pickle=False).astype(np.float32)
        if sequence.ndim != 2 or sequence.shape != (self.expected_sequence_length, 128):
            raise ValueError(f"Expected landmark sequence shape [{self.expected_sequence_length}, 128], got {sequence.shape}")
        if not np.isfinite(sequence).all():
            raise ValueError(f"Landmark sequence contains non-finite values: {self.sequence_paths[index]}")
        if self.transform is not None:
            sequence = self.transform(sequence)
            if sequence.shape != (self.expected_sequence_length, 128) or not np.isfinite(sequence).all():
                raise ValueError("Landmark transform returned an invalid sequence")
        return torch.from_numpy(sequence), torch.tensor(self.labels[index], dtype=torch.long)


__all__ = ["LandmarkSequenceDataset"]
