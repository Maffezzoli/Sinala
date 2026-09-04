from pathlib import Path

import pytest

from sinala.cli import _build_training_samples, _landmark_cache_key


def test_build_training_samples_when_path_is_duplicated_then_rejects_manifest() -> None:
    entries = [
        {"sequence_path": "a.npy", "label": "Acontecer", "signer_id": "01"},
        {"sequence_path": "a.npy", "label": "Acontecer", "signer_id": "02"},
    ]

    with pytest.raises(ValueError, match="duplicate sequence path"):
        _build_training_samples(entries, {"Acontecer"}, {"01", "02"})


def test_build_training_samples_when_signer_is_unassigned_then_rejects_manifest() -> None:
    entries = [{"sequence_path": "a.npy", "label": "Acontecer", "signer_id": "03"}]

    with pytest.raises(ValueError, match="not assigned"):
        _build_training_samples(entries, {"Acontecer"}, {"01", "02"})


def test_landmark_cache_key_when_source_changes_then_changes_key(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    model_path = tmp_path / "model.task"
    video_path.write_bytes(b"one")
    model_path.write_bytes(b"model")

    first_key = _landmark_cache_key(video_path, model_path, 32)
    video_path.write_bytes(b"two")

    assert _landmark_cache_key(video_path, model_path, 32) != first_key
