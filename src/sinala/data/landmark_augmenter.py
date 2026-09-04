from collections.abc import Callable

import numpy as np


class LandmarkAugmenter:
    def __init__(
        self,
        probability: float = 0.8,
        noise_std: float = 0.01,
        scale_range: tuple[float, float] = (0.9, 1.1),
        temporal_speed_range: tuple[float, float] = (0.85, 1.15),
        seed: int = 42,
    ) -> None:
        numeric_parameters = np.asarray([probability, noise_std, *scale_range, *temporal_speed_range], dtype=np.float64)
        if not np.isfinite(numeric_parameters).all():
            raise ValueError("augmentation parameters must be finite")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if scale_range[0] <= 0.0 or scale_range[0] > scale_range[1]:
            raise ValueError("scale_range must contain positive ascending values")
        if temporal_speed_range[0] <= 0.0 or temporal_speed_range[0] > temporal_speed_range[1]:
            raise ValueError("temporal_speed_range must contain positive ascending values")
        self.probability = probability
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.temporal_speed_range = temporal_speed_range
        self._random = np.random.default_rng(seed)

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            augmented = sequence.astype(np.float32, copy=True)
        self._validate(augmented)
        if self._random.random() >= self.probability:
            return augmented
        with np.errstate(over="ignore", invalid="ignore"):
            augmented = self._augment_spatial(augmented)
            augmented = self._augment_temporal(augmented)
        self._validate(augmented)
        return augmented

    @staticmethod
    def _validate(sequence: np.ndarray) -> None:
        if sequence.ndim != 2 or sequence.shape[0] < 1 or sequence.shape[1] != 128:
            raise ValueError(f"Expected landmark sequence shape [time>=1, 128], got {sequence.shape}")
        if not np.isfinite(sequence).all():
            raise ValueError("Landmark sequence contains non-finite values")

    def _augment_spatial(self, sequence: np.ndarray) -> np.ndarray:
        scale = self._random.uniform(*self.scale_range)
        translation = self._random.normal(0.0, self.noise_std, size=(1, 1, 3)).astype(np.float32)
        for start in (0, 64):
            coordinates = sequence[:, start : start + 63].reshape(sequence.shape[0], 21, 3)
            present = sequence[:, start + 63 : start + 64] > 0.5
            coordinates *= scale
            coordinates += translation * present[:, :, None]
            if self.noise_std:
                coordinates += self._random.normal(0.0, self.noise_std, size=coordinates.shape).astype(np.float32) * present[:, :, None]
            sequence[:, start : start + 63] = coordinates.reshape(sequence.shape[0], 63)
        return sequence

    def _augment_temporal(self, sequence: np.ndarray) -> np.ndarray:
        speed = self._random.uniform(*self.temporal_speed_range)
        time = np.arange(sequence.shape[0], dtype=np.float32)
        center = (sequence.shape[0] - 1) / 2.0
        source_time = np.clip(center + (time - center) * speed, 0.0, center * 2.0)
        result = np.empty_like(sequence)
        for feature_index in range(128):
            result[:, feature_index] = np.interp(source_time, time, sequence[:, feature_index])
        nearest = np.clip(np.rint(source_time).astype(np.int64), 0, sequence.shape[0] - 1)
        for start, mask_index in ((0, 63), (64, 127)):
            result[:, mask_index] = sequence[nearest, mask_index]
            result[:, start : start + 63] *= result[:, mask_index, None]
        return result


Augmentation = Callable[[np.ndarray], np.ndarray]


__all__ = ["Augmentation", "LandmarkAugmenter"]
