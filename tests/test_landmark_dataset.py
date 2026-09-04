from pathlib import Path

import numpy as np
import pytest

from sinala.data.landmark_dataset import LandmarkSequenceDataset


def test_landmark_sequence_dataset_when_sequence_length_differs_then_rejects_sample(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence.npy"
    np.save(sequence_path, np.zeros((16, 128), dtype=np.float32), allow_pickle=False)
    dataset = LandmarkSequenceDataset([sequence_path], [0], expected_sequence_length=32)

    with pytest.raises(ValueError, match="Expected landmark sequence shape"):
        dataset[0]
