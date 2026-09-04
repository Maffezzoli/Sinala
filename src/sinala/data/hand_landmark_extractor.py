from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class HandLandmarkExtractor:
    FEATURE_SIZE = 128
    LANDMARKS_PER_HAND = 21

    def __init__(self, model_asset_path: Path, assume_mirrored_input: bool = False) -> None:
        if not model_asset_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand model does not exist: {model_asset_path}")
        self.model_asset_path = model_asset_path
        self.assume_mirrored_input = assume_mirrored_input
        self._landmarker = self._create_landmarker()

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, Any]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        results = self._landmarker.detect(image)
        hand_points = self._collect_hands(results)
        return self._normalize_hands(hand_points), results

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        features, _ = self.process(frame_bgr)
        return features

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        _, results = self.process(frame_bgr)
        return self.draw_results(frame_bgr, results)

    def draw_results(self, frame_bgr: np.ndarray, results: Any) -> np.ndarray:
        for hand_landmarks in results.hand_landmarks or []:
            points = [(int(landmark.x * frame_bgr.shape[1]), int(landmark.y * frame_bgr.shape[0])) for landmark in hand_landmarks]
            for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                cv2.line(frame_bgr, points[connection.start], points[connection.end], (0, 82, 255), 2, cv2.LINE_AA)
            for x, y in points:
                cv2.circle(frame_bgr, (x, y), 4, (0, 82, 255), -1, cv2.LINE_AA)
        return frame_bgr

    def close(self) -> None:
        self._landmarker.close()

    def reset(self) -> None:
        self.close()
        self._landmarker = self._create_landmarker()

    def _create_landmarker(self) -> Any:
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_asset_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _collect_hands(self, results: Any) -> dict[str, np.ndarray]:
        hands: dict[str, np.ndarray] = {}
        landmarks = results.hand_landmarks or []
        handedness = results.handedness or []
        for index, hand_landmarks in enumerate(landmarks):
            label = "unknown"
            if index < len(handedness) and handedness[index]:
                label = handedness[index][0].category_name.casefold()
            if self.assume_mirrored_input:
                label = {"left": "right", "right": "left"}.get(label, label)
            points = np.array([[point.x, point.y, point.z] for point in hand_landmarks], dtype=np.float32)
            if label not in {"left", "right"} or label in hands:
                label = "left" if "left" not in hands else "right"
            hands[label] = points
        return hands

    def _normalize_hands(self, hands: dict[str, np.ndarray]) -> np.ndarray:
        present_hands = [points for points in hands.values() if len(points)]
        if present_hands:
            wrists = np.array([points[0] for points in present_hands], dtype=np.float32)
            center = wrists.mean(axis=0)
            all_points = np.concatenate(present_hands, axis=0)
            scale = float(np.linalg.norm(all_points - center, axis=1).max())
        else:
            center = np.zeros(3, dtype=np.float32)
            scale = 1.0
        scale = max(scale, 1e-6)

        features: list[float] = []
        for label in ("left", "right"):
            points = hands.get(label)
            if points is None:
                features.extend([0.0] * (self.LANDMARKS_PER_HAND * 3))
                features.append(0.0)
                continue
            normalized = (points - center) / scale
            features.extend(normalized.astype(np.float32).reshape(-1).tolist())
            features.append(1.0)
        return np.asarray(features, dtype=np.float32)


__all__ = ["HandLandmarkExtractor"]
