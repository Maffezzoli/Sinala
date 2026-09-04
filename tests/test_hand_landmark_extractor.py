import numpy as np

from sinala.data.hand_landmark_extractor import HandLandmarkExtractor


def test_hand_landmark_extractor_when_one_hand_is_missing_then_returns_masked_features() -> None:
    extractor = object.__new__(HandLandmarkExtractor)
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0.0, 1.0, 21)

    features = extractor._normalize_hands({"left": points})

    assert features.shape == (128,)
    assert features[63] == 1.0
    assert features[64:127].tolist() == [0.0] * 63
    assert features[127] == 0.0


def test_hand_landmark_extractor_when_no_hands_are_detected_then_returns_zero_features() -> None:
    extractor = object.__new__(HandLandmarkExtractor)

    features = extractor._normalize_hands({})

    assert features.shape == (128,)
    assert np.array_equal(features, np.zeros(128, dtype=np.float32))
