import numpy as np
import pytest

from sinala.data.landmark_augmenter import LandmarkAugmenter


def test_landmark_augmenter_when_probability_is_zero_then_preserves_values() -> None:
    sequence = np.arange(32 * 128, dtype=np.float32).reshape(32, 128)
    augmenter = LandmarkAugmenter(probability=0.0)

    result = augmenter(sequence)

    assert np.array_equal(result, sequence)
    assert result is not sequence


def test_landmark_augmenter_when_enabled_then_preserves_shape_masks_and_finite_values() -> None:
    sequence = np.zeros((32, 128), dtype=np.float32)
    sequence[:, :63] = 0.25
    sequence[:, 63] = 1.0
    sequence[:, 64:127] = -0.25
    sequence[:, 127] = 1.0
    augmenter = LandmarkAugmenter(probability=1.0, noise_std=0.01, seed=7)

    result = augmenter(sequence)

    assert result.shape == sequence.shape
    assert np.isfinite(result).all()
    assert np.array_equal(result[:, 63], sequence[:, 63])
    assert np.array_equal(result[:, 127], sequence[:, 127])
    assert not np.array_equal(result[:, :63], sequence[:, :63])


def test_landmark_augmenter_when_hand_disappears_then_zeroes_interpolated_coordinates() -> None:
    sequence = np.zeros((32, 128), dtype=np.float32)
    sequence[8:, :63] = 1.0
    sequence[8:, 63] = 1.0
    augmenter = LandmarkAugmenter(probability=1.0, noise_std=0.0, seed=7)

    result = augmenter(sequence)

    assert np.all(result[result[:, 63] == 0.0, :63] == 0.0)


def test_landmark_augmenter_when_parameters_are_nonfinite_then_rejects_configuration() -> None:
    with pytest.raises(ValueError, match="parameters must be finite"):
        LandmarkAugmenter(noise_std=float("nan"))


def test_landmark_augmenter_when_sequence_has_no_frames_then_rejects_input() -> None:
    with pytest.raises(ValueError, match="time>=1"):
        LandmarkAugmenter()(np.zeros((0, 128), dtype=np.float32))


def test_landmark_augmenter_when_float64_cast_overflows_then_rejects_input() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        LandmarkAugmenter(probability=0.0)(np.full((32, 128), 1e39, dtype=np.float64))


def test_landmark_augmenter_when_parameters_overflow_float32_then_rejects_result() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        LandmarkAugmenter(probability=1.0, noise_std=1e39)(np.ones((32, 128), dtype=np.float32))


def test_landmark_augmenter_when_sequence_shape_is_invalid_then_rejects_input() -> None:
    with pytest.raises(ValueError, match="Expected landmark sequence shape"):
        LandmarkAugmenter()(np.zeros((32, 64), dtype=np.float32))
