from collections import deque
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from sinala.data.hand_landmark_extractor import HandLandmarkExtractor
from sinala.model.gru_sign_classifier import GRUSignClassifier
from sinala.model.training import load_checkpoint
from sinala.runtime.stable_caption_buffer import StableCaptionBuffer


class RealtimeCaptionApp:
    def __init__(
        self,
        checkpoint_path: Path,
        model_asset_path: Path,
        camera_index: int = 0,
        sequence_length: int = 32,
        confidence_threshold: float = 0.75,
        stability_window_size: int = 5,
        minimum_consensus: int = 4,
        cooldown_seconds: float = 0.8,
        device: torch.device | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.model_asset_path = model_asset_path
        self.camera_index = camera_index
        self.sequence_length = sequence_length
        self.device = device or torch.device("cpu")
        self.model, self.payload = self._load_model()
        self.extractor = HandLandmarkExtractor(model_asset_path)
        self.caption_buffer = StableCaptionBuffer(
            confidence_threshold=confidence_threshold,
            window_size=stability_window_size,
            minimum_consensus=minimum_consensus,
            cooldown_seconds=cooldown_seconds,
        )

    def run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        frames: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        last_prediction = "aguardando sinal"
        last_confidence = 0.0
        previous_time = perf_counter()
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Could not read frame from camera")
                features, results = self.extractor.process(frame)
                frames.append(features)
                hands_detected = bool(features[-65] or features[-1])
                if len(frames) == self.sequence_length:
                    label, confidence = self._predict(np.stack(frames))
                    stable_label = self.caption_buffer.update(label, confidence, hands_detected)
                    if stable_label is not None:
                        last_prediction = stable_label
                    last_confidence = confidence
                self.extractor.draw_results(frame, results)
                current_time = perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time
                display = cv2.flip(frame, 1)
                self._draw_status(display, last_prediction, last_confidence, fps, hands_detected)
                cv2.imshow("SINALA", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            self.extractor.close()
            cv2.destroyAllWindows()

    def _load_model(self) -> tuple[GRUSignClassifier, dict[str, object]]:
        model, payload = load_checkpoint(self.checkpoint_path, self.device)
        metadata = payload["metadata"]
        if not isinstance(metadata, dict) or metadata.get("sequence_length") != self.sequence_length or metadata.get("feature_size") != 128:
            raise ValueError("Checkpoint metadata does not match the camera feature and sequence schema")
        index_to_class = payload["index_to_class"]
        expected_indices = {str(index) for index in range(model.number_of_classes)}
        if not isinstance(index_to_class, dict) or set(index_to_class) != expected_indices or len(set(index_to_class.values())) != model.number_of_classes:
            raise ValueError("Checkpoint class mapping is invalid")
        return model, payload

    def _predict(self, sequence: np.ndarray) -> tuple[str, float]:
        tensor = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        class_index = int(probabilities.argmax().item())
        confidence = float(probabilities[class_index].item())
        index_to_class = self.payload["index_to_class"]
        return str(index_to_class[str(class_index)]), confidence

    @staticmethod
    def _draw_status(frame: np.ndarray, label: str, confidence: float, fps: float, hands_detected: bool) -> None:
        cv2.putText(frame, f"Legenda: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 25, 47), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confianca: {confidence:.2f}  FPS: {fps:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 25, 47), 2, cv2.LINE_AA)
        status = "maos detectadas" if hands_detected else "nenhuma mao detectada"
        cv2.putText(frame, status, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 82, 255), 2, cv2.LINE_AA)


__all__ = ["RealtimeCaptionApp"]
