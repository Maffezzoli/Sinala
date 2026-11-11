"""Dataset, augmentation, and geometric disambiguation for the Libras fingerspelling alphabet."""

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

_FINGER_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
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


def _finger_state(pts: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> tuple[bool, bool]:
    seg_sum = float(np.linalg.norm(pts[pip] - pts[mcp])) + float(np.linalg.norm(pts[dip] - pts[pip])) + float(np.linalg.norm(pts[tip] - pts[dip]))
    ratio = float(np.linalg.norm(pts[tip] - pts[mcp])) / max(seg_sum, 1e-6)
    r_mcp = float(np.linalg.norm(pts[mcp]))
    r_tip = float(np.linalg.norm(pts[tip]))
    is_extended = ratio > 0.75 and r_tip > r_mcp * 1.20
    is_folded = ratio < 0.68 or r_tip < r_mcp * 1.15
    return is_extended, is_folded


def refine_alphabet_prediction(
    coords: np.ndarray,
    predicted_letter: str,
    confidence: float,
) -> tuple[str, float]:
    """Disambiguate minimal-pair Libras signs (U/V/R/N, M/W, G/L/Q) using biomechanical invariants."""
    pts_raw = np.asarray(coords, dtype=np.float32).reshape(LANDMARK_COUNT, 3)
    rel = pts_raw - pts_raw[0:1]
    scale = float(np.linalg.norm(rel[9]))
    if scale < 1e-6:
        return predicted_letter, confidence
    p = rel / scale

    idx_ext, _ = _finger_state(p, 5, 6, 7, 8)
    mid_ext, mid_fold = _finger_state(p, 9, 10, 11, 12)
    rng_ext, rng_fold = _finger_state(p, 13, 14, 15, 16)
    _, pnk_fold = _finger_state(p, 17, 18, 19, 20)

    # 1. Two-finger family: Index & Middle extended, Ring & Pinky folded
    if idx_ext and mid_ext and rng_fold and pnk_fold:
        dy_idx = float(p[8, 1] - p[5, 1])
        dy_mid = float(p[12, 1] - p[9, 1])
        if dy_idx > 0.45 and dy_mid > 0.45:
            return "N", max(confidence, 0.85)
        # Ensure fingertips are genuinely raised upright (excludes curved O/C)
        if dy_idx < -0.55 and dy_mid < -0.55 and float(p[8, 1]) < -1.35 and float(p[12, 1]) < -1.35:
            d_thumb_pip = min(float(np.linalg.norm(p[4] - p[6])), float(np.linalg.norm(p[4] - p[10])))
            if predicted_letter in ("K", "H") and d_thumb_pip < 0.36:
                return predicted_letter, confidence
            mcp_dir = 1.0 if (p[5, 0] - p[9, 0]) >= 0 else -1.0
            signed_sep = float(p[8, 0] - p[12, 0]) * mcp_dir
            d_xy = float(np.hypot(p[8, 0] - p[12, 0], p[8, 1] - p[12, 1]))
            v1 = p[8, :2] - p[5, :2]
            v2 = p[12, :2] - p[9, :2]
            cos_a = float(np.dot(v1, v2)) / max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-6)
            ang_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            if signed_sep < -0.02:
                return "R", max(confidence, 0.85)
            if d_xy >= 0.24 or ang_deg >= 7.2:
                return "V", max(confidence, 0.85)
            return "U", max(confidence, 0.85)

    # 2. Three-finger family: Index, Middle, Ring extended, Pinky folded -> M (down) vs W (up)
    if idx_ext and mid_ext and rng_ext and pnk_fold:
        if float(p[8, 1] - p[5, 1]) > 0.45 and float(p[12, 1] - p[9, 1]) > 0.45:
            return "M", max(confidence, 0.85)
        if float(p[8, 1] - p[5, 1]) < -0.45 and float(p[12, 1] - p[9, 1]) < -0.45:
            return "W", max(confidence, 0.85)

    # 3. Index-only family: Q (pointing down) vs L (thumb 90 deg)
    if idx_ext and mid_fold and rng_fold and pnk_fold:
        if float(p[8, 1] - p[5, 1]) > 0.55:
            return "Q", max(confidence, 0.85)
        if float(np.linalg.norm(p[4, :2] - p[6, :2])) > 0.75:
            return "L", max(confidence, 0.85)

    return predicted_letter, confidence


def _rotate_2d_finger(pts: np.ndarray, mcp: int, joints: tuple[int, ...], angle_rad: np.ndarray) -> None:
    cos_a = np.cos(angle_rad)[:, None]
    sin_a = np.sin(angle_rad)[:, None]
    anchor = pts[:, mcp : mcp + 1, :2]
    rel = pts[:, joints, :2] - anchor
    rx = rel[:, :, 0] * cos_a - rel[:, :, 1] * sin_a
    ry = rel[:, :, 0] * sin_a + rel[:, :, 1] * cos_a
    pts[:, joints, 0] = anchor[:, :, 0] + rx
    pts[:, joints, 1] = anchor[:, :, 1] + ry


def _augment_class_landmarks(
    letter: str,
    raw_train_by_letter: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate cross-signer & cross-camera domain-randomized landmarks for `letter`."""
    base = raw_train_by_letter[letter].copy()  # (N, 21, 3)
    n = len(base)
    aug1 = base.copy()
    aug2 = base.copy()

    for copy_arr in (aug1, aug2):
        rel = copy_arr - copy_arr[:, 0:1, :]
        sc = np.linalg.norm(rel[:, 9, :], axis=-1, keepdims=True)
        sc = np.where(sc < 1e-6, 1.0, sc)

        # Jitter folded (non-defining) fingers so MLP never memorizes one person's fist curl
        for mcp, pip, dip, tip in _FINGER_CHAINS:
            seg = (
                np.linalg.norm(rel[:, pip] - rel[:, mcp], axis=1)
                + np.linalg.norm(rel[:, dip] - rel[:, pip], axis=1)
                + np.linalg.norm(rel[:, tip] - rel[:, dip], axis=1)
            )
            ext = np.linalg.norm(rel[:, tip] - rel[:, mcp], axis=1) / np.maximum(seg, 1e-6)
            folded_mask = ext < 0.68
            if np.any(folded_mask):
                curl_jitter = rng.normal(0.0, 0.08, size=(int(folded_mask.sum()), 3, 3)).astype(np.float32)
                rel[folded_mask, pip : tip + 1, :] += curl_jitter * sc[folded_mask, :, None]

        # Camera Z-depth scale & webcam aspect-ratio (4:3 vs 16:9) variation
        rel[:, :, 2] *= rng.uniform(0.35, 1.45, size=(n, 1)).astype(np.float32)
        rel[:, :, 0] *= rng.uniform(0.78, 1.22, size=(n, 1)).astype(np.float32)
        rel += rng.normal(0.0, 0.008, size=rel.shape).astype(np.float32) * sc[:, :, None]
        copy_arr[:] = copy_arr[:, 0:1, :] + rel

    extras = [base, aug1, aug2]

    # Explicit cross-fist decoupling for U vs V vs R so MLP relies ONLY on index-middle geometry
    if letter in ("U", "V", "R") and all(k in raw_train_by_letter for k in ("U", "V", "R")):
        donors = np.concatenate([raw_train_by_letter["U"], raw_train_by_letter["V"], raw_train_by_letter["R"]], axis=0)
        donor_idx = rng.integers(0, len(donors), size=n)
        synth = donors[donor_idx].copy()
        d_rel = synth - synth[:, 0:1, :]
        d_sc = np.maximum(np.linalg.norm(d_rel[:, 9, :], axis=-1, keepdims=True), 1e-6)

        src_rel = base - base[:, 0:1, :]
        src_sc = np.maximum(np.linalg.norm(src_rel[:, 9, :], axis=-1, keepdims=True), 1e-6)
        # Transplant normalized index+middle (5..12) onto diverse donor palm/thumb/ring/pinky!
        d_rel[:, 5:13, :] = (src_rel[:, 5:13, :] / src_sc[:, :, None]) * d_sc[:, :, None]
        synth = synth[:, 0:1, :] + d_rel

        if letter == "V":
            # Also cover modest-to-wide human V angles (8 deg to 28 deg total spread)
            u_donors = raw_train_by_letter["U"][rng.integers(0, len(raw_train_by_letter["U"]), size=n)].copy()
            mcp_sign = np.where((u_donors[:, 5, 0] - u_donors[:, 9, 0]) >= 0, 1.0, -1.0).astype(np.float32)
            half_spread = np.radians(rng.uniform(4.5, 14.0, size=n)).astype(np.float32)
            _rotate_2d_finger(u_donors, 5, (6, 7, 8), half_spread * mcp_sign)
            _rotate_2d_finger(u_donors, 9, (10, 11, 12), -half_spread * mcp_sign)
            extras.extend([synth, u_donors])
        else:
            extras.append(synth)

    return np.concatenate(extras, axis=0).reshape(-1, LANDMARK_COUNT * 3)


class LibrasAlphabetDataset(Dataset):
    """PyTorch Dataset for the 26-letter Libras Alphabet."""

    CLASSES: ClassVar[tuple[str, ...]] = ALPHABET_CLASSES

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        augment: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}

        raw_chosen_by_letter: dict[str, np.ndarray] = {}
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
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(arr))
            split_point = int(len(arr) * train_ratio)

            if split == "train":
                chosen_idx = perm[:split_point]
            elif split in ("test", "val"):
                chosen_idx = perm[split_point:]
            else:
                chosen_idx = perm

            raw_chosen_by_letter[letter] = arr[chosen_idx].reshape(-1, LANDMARK_COUNT, 3)

        all_samples: list[np.ndarray] = []
        all_labels: list[int] = []
        aug_rng = np.random.default_rng(seed + 101)

        for letter in self.CLASSES:
            if letter not in raw_chosen_by_letter:
                continue
            if split == "train" and augment:
                coords_batch = _augment_class_landmarks(letter, raw_chosen_by_letter, aug_rng)
            else:
                coords_batch = raw_chosen_by_letter[letter].reshape(-1, LANDMARK_COUNT * 3)

            norm_samples = compute_rotation_invariant_features(coords_batch)
            if norm_samples.ndim == 1:
                norm_samples = norm_samples[None, :]
            all_samples.append(norm_samples)
            all_labels.extend([self.class_to_idx[letter]] * len(norm_samples))

        self.features = np.vstack(all_samples).astype(np.float32)
        self.targets = np.array(all_labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.tensor(self.targets[idx], dtype=torch.long)
