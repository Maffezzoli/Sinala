from pathlib import Path

import cv2
import numpy as np

from sinala.data.hand_landmark_extractor import HandLandmarkExtractor


class LandmarkPreprocessor:
    def __init__(self, extractor: HandLandmarkExtractor, sequence_length: int = 32) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.extractor = extractor
        self.sequence_length = sequence_length

    def process_video(self, video_path: Path) -> np.ndarray:
        self.extractor.reset()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        target_indices = self._target_indices(frame_count)
        selected: list[np.ndarray] = []
        frame_index = 0
        target_position = 0
        try:
            while target_position < len(target_indices):
                success, frame = capture.read()
                if not success:
                    break
                if frame_index == int(target_indices[target_position]):
                    selected.append(self.extractor.extract(frame))
                    target_position += 1
                frame_index += 1
        finally:
            capture.release()

        if not selected:
            raise ValueError(f"Video contains no readable frames: {video_path}")
        while len(selected) < self.sequence_length:
            selected.append(selected[-1].copy())
        return np.stack(selected[: self.sequence_length]).astype(np.float32)

    def _target_indices(self, frame_count: int) -> np.ndarray:
        if frame_count <= 0:
            return np.arange(self.sequence_length, dtype=np.int64)
        return np.unique(np.linspace(0, max(frame_count - 1, 0), self.sequence_length, dtype=np.int64))


__all__ = ["LandmarkPreprocessor"]
