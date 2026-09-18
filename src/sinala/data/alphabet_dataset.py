"""Dataset and geometric normalization for the Libras fingerspelling alphabet."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar

import kagglehub
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

ALPHABET_CLASSES: tuple[str, ...] = tuple("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split())


def download_alphabet_dataset(target_dir: str | Path = "data/raw/alphabet") -> Path:
    """Download the open Libras Alphabet Landmark dataset via kagglehub."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    existing_classes = {path.stem for path in target_path.glob("*.csv")}
    if existing_classes == set(ALPHABET_CLASSES):
        logger.info(f"Dataset do alfabeto já presente em {target_path} ({len(existing_classes)} classes)")
        return target_path

    logger.info("Baixando dataset heitorccf/librasign via kagglehub...")
    downloaded_dir = Path(kagglehub.dataset_download("heitorccf/librasign"))
    source_landmarks = downloaded_dir / "landmarks"
    if not source_landmarks.exists():
        source_landmarks = downloaded_dir

    copied = 0
    for letter in ALPHABET_CLASSES:
        src_file = source_landmarks / f"{letter}.csv"
        if src_file.exists():
            dest_file = target_path / f"{letter}.csv"
            dest_file.write_bytes(src_file.read_bytes())
            copied += 1

    logger.info(f"Dataset do alfabeto sincronizado com sucesso: {copied} classes em {target_path}")
    return target_path


LANDMARK_COUNT: int = 21
FEATURE_DIM_ALPHABET: int = 225

PAIRWISE_INDEXES: tuple[tuple[int, int], ...] = tuple((i, j) for i in range(LANDMARK_COUNT) for j in range(i + 1, LANDMARK_COUNT))

ANGLE_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    (1, 2, 3),
    (2, 3, 4),
    (0, 1, 2),
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
)


def compute_rotation_invariant_features(coords: np.ndarray) -> np.ndarray:
    """Compute 225 features invariant to translation, scale, rotation and reflection."""
    arr = np.asarray(coords, dtype=np.float32).reshape(-1, LANDMARK_COUNT, 3).copy()
    wrist = arr[:, 0:1, :]
    relative = arr - wrist
    scale = np.linalg.norm(relative[:, 9, :], axis=-1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = relative / scale[:, :, None]

    point_a = [a for a, _ in PAIRWISE_INDEXES]
    point_b = [b for _, b in PAIRWISE_INDEXES]
    distances = np.linalg.norm(normalized[:, point_a, :] - normalized[:, point_b, :], axis=2)

    angles = []
    for a, b, c in ANGLE_TRIPLETS:
        first = normalized[:, a, :] - normalized[:, b, :]
        second = normalized[:, c, :] - normalized[:, b, :]
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        denominator = np.where(denominator < 1e-6, 1.0, denominator)
        cosine = np.clip(np.sum(first * second, axis=1) / denominator, -1.0, 1.0)
        angles.append(np.arccos(cosine))

    features = np.hstack([distances, np.column_stack(angles)]).astype(np.float32)
    if coords.ndim == 1 or (coords.ndim == 2 and coords.shape == (LANDMARK_COUNT, 3)):
        return features[0]
    return features


def normalize_hand_landmarks(coords: np.ndarray, is_left_hand: bool = True) -> np.ndarray:
    """Extract 225 rotation-invariant geometric features from hand landmarks."""
    return compute_rotation_invariant_features(coords)


class LibrasAlphabetDataset(Dataset):
    """PyTorch Dataset for the 26-letter Libras Alphabet."""

    CLASSES: ClassVar[tuple[str, ...]] = ALPHABET_CLASSES

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}

        all_samples: list[np.ndarray] = []
        all_labels: list[int] = []

        for letter in self.CLASSES:
            csv_path = self.data_dir / f"{letter}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Arquivo de classe ausente: {csv_path}")

            class_samples: list[list[float]] = []
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        class_samples.append([float(val) for val in row])

            if not class_samples:
                continue

            arr = np.array(class_samples, dtype=np.float32)
            # Normalizar todas as amostras
            norm_samples = np.zeros((len(arr), FEATURE_DIM_ALPHABET), dtype=np.float32)
            for i in range(len(arr)):
                norm_samples[i] = normalize_hand_landmarks(arr[i])

            # Split estratificado por classe
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(norm_samples))
            split_point = int(len(norm_samples) * train_ratio)

            if split == "train":
                chosen_idx = perm[:split_point]
            elif split in ("test", "val"):
                chosen_idx = perm[split_point:]
            else:
                chosen_idx = perm

            all_samples.append(norm_samples[chosen_idx])
            all_labels.extend([self.class_to_idx[letter]] * len(chosen_idx))

        self.features = np.vstack(all_samples).astype(np.float32)
        self.targets = np.array(all_labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.tensor(self.targets[idx], dtype=torch.long)
