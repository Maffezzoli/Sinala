from pathlib import Path

import pytest
from pydantic import ValidationError

from sinala.config.pipeline_settings import PipelineSettings
from sinala.data.catalog import canonicalize_label, catalog_dataset


def test_pipeline_settings_when_defaults_are_valid_then_loads_configuration() -> None:
    settings = PipelineSettings()

    assert settings.sequence_length == 32
    assert settings.confidence_threshold == 0.75


def test_pipeline_settings_when_consensus_exceeds_window_then_rejects_configuration() -> None:
    with pytest.raises(ValidationError):
        PipelineSettings(stability_window_size=3, minimum_consensus=4)


def test_pipeline_settings_when_random_seed_is_negative_then_rejects_configuration() -> None:
    with pytest.raises(ValidationError):
        PipelineSettings(random_seed=-1)


def test_canonicalize_label_when_case_and_accents_differ_then_returns_official_label() -> None:
    assert canonicalize_label("maca") == "Maçã"


def test_catalog_when_valid_dataset_tree_then_builds_manifest(tmp_path: Path) -> None:
    video_path = tmp_path / "Sinalizador01" / "Acontecer" / "video_01.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    samples = catalog_dataset(tmp_path)

    assert len(samples) == 1
    assert samples[0].label == "Acontecer"
    assert samples[0].signer_id == "01"


def test_catalog_when_filename_is_unrecognized_then_ignores_sample(tmp_path: Path) -> None:
    video_path = tmp_path / "unknown" / "video.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    assert catalog_dataset(tmp_path) == []
