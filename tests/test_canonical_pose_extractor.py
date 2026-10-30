"""Unit tests for the CanonicalPoseExtractor utility."""

import json
from pathlib import Path

import numpy as np

from sinala.model.canonical_pose_extractor import CanonicalPoseExtractor


def test_canonical_pose_extractor_when_valid_csvs_then_exports_json(tmp_path: Path) -> None:
    csv_dir = tmp_path / "alphabet"
    csv_dir.mkdir(parents=True)
    empty_greetings = tmp_path / "empty_greetings"
    empty_greetings.mkdir()

    dummy_points = np.random.uniform(0.1, 0.9, size=(5, 63)).astype(np.float32)
    np.savetxt(csv_dir / "A.csv", dummy_points, delimiter=",")

    output_dir = tmp_path / "web_data"
    extractor = CanonicalPoseExtractor(raw_alphabet_dir=csv_dir, greetings_dir=empty_greetings, output_dir=output_dir)

    output_path = extractor.extract_and_export(filename="poses.json")

    assert output_path.exists()
    content = json.loads(output_path.read_text(encoding="utf-8"))
    assert "A" in content
    assert len(content["A"]["landmarks"]) == 21
    assert len(content["A"]["landmarks"][0]) == 3


def test_canonical_pose_extractor_when_empty_dir_then_exports_empty_json(tmp_path: Path) -> None:
    empty_csv_dir = tmp_path / "empty_alphabet"
    empty_csv_dir.mkdir()
    empty_greetings = tmp_path / "empty_greetings"
    empty_greetings.mkdir()
    empty_videos = tmp_path / "empty_videos"
    empty_videos.mkdir()

    output_dir = tmp_path / "web_data"
    extractor = CanonicalPoseExtractor(
        raw_alphabet_dir=empty_csv_dir,
        greetings_dir=empty_greetings,
        raw_greetings_video_dir=empty_videos,
        output_dir=output_dir,
    )
    output_path = extractor.extract_and_export(filename="empty_poses.json")
    content = json.loads(output_path.read_text(encoding="utf-8"))
    assert content == {}
