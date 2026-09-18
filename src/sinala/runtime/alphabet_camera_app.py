"""Real-time camera application for generic Libras fingerspelling transcription."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger

from sinala.data.alphabet_dataset import normalize_hand_landmarks
from sinala.data.hand_landmark_extractor import HandLandmarkExtractor
from sinala.model.alphabet_classifier import AlphabetClassifier, load_alphabet_checkpoint
from sinala.runtime.word_builder import WordBuilder


class AlphabetCameraApp:
    """Generic real-time transcription app for Libras manual alphabet."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        model_asset_path: Path | str,
        camera_index: int = 0,
        confidence_threshold: float = 0.70,
        stability_frames: int = 12,
        device: torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model_asset_path = Path(model_asset_path)
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device("cpu")

        logger.info(f"Carregando classificador de dactilologia de {self.checkpoint_path}...")
        self.model: AlphabetClassifier = load_alphabet_checkpoint(self.checkpoint_path, device=str(self.device))
        self.extractor = HandLandmarkExtractor(self.model_asset_path)
        self.word_builder = WordBuilder(
            stability_frames=stability_frames,
            confidence_threshold=confidence_threshold,
        )
        self.primary_handedness: str | None = None

    def run(self) -> None:
        """Run the interactive webcam transcription loop."""
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera índice {self.camera_index}")

        logger.info("Transcrição de dactilologia iniciada. Pressione 'q' para sair.")
        previous_time = perf_counter()

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Falha ao capturar quadro da webcam.")
                features, results = self.extractor.process(frame)

                detected_letter, confidence = self._process_hand(results)

                # Atualiza buffer genérico de palavras
                self.word_builder.update(detected_letter, confidence)

                # Desenha landmarks na imagem
                self.extractor.draw_results(frame, results)

                current_time = perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time

                display = cv2.flip(frame, 1)
                self._draw_hud(display, detected_letter, confidence, fps)

                cv2.imshow("SINALA - Dactilologia Libras", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == 32:  # Barra de espaço
                    self.word_builder.add_space()
                elif key in (8, 127):  # Backspace
                    self.word_builder.backspace()
                elif key == ord("c"):  # Limpar
                    self.word_builder.clear()

        finally:
            capture.release()
            self.extractor.close()
            cv2.destroyAllWindows()

    def _process_hand(self, results: Any) -> tuple[str, float]:
        """Extract landmarks from the primary hand and predict letter."""
        landmarks = results.hand_landmarks or []
        handedness = results.handedness or []
        if not landmarks:
            self.primary_handedness = None
            return "?", 0.0

        hand_index = 0
        if self.primary_handedness and handedness:
            for index, candidates in enumerate(handedness):
                if candidates and candidates[0].category_name.casefold() == self.primary_handedness:
                    hand_index = index
                    break

        primary_landmarks = landmarks[hand_index]
        is_left_hand = False
        if handedness and hand_index < len(handedness) and handedness[hand_index]:
            label = handedness[hand_index][0].category_name.casefold()
            self.primary_handedness = label
            is_left_hand = label == "left"

        coords = np.array([[lm.x, lm.y, lm.z] for lm in primary_landmarks], dtype=np.float32)
        norm_coords = normalize_hand_landmarks(coords, is_left_hand=is_left_hand)

        tensor_in = torch.from_numpy(norm_coords).to(self.device)
        return self.model.predict_letter(tensor_in, threshold=self.confidence_threshold)

    def _draw_hud(self, frame: np.ndarray, letter: str, conf: float, fps: float) -> None:
        """Render minimal, generic transcription HUD overlay."""
        h, w, _ = frame.shape

        # Barra de status superior e inferior discretas
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (10, 25, 47), -1)
        cv2.rectangle(overlay, (0, h - 75), (w, h), (10, 25, 47), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Status da letra detectada
        progress = self.word_builder.get_progress()
        prog_bars = int(progress * 10)
        prog_str = "[" + "=" * prog_bars + " " * (10 - prog_bars) + "]"

        status_letter = f"LETRA: {letter} ({conf * 100:.0f}%) {prog_str}" if letter != "?" else "AGUARDANDO SINAL"
        cv2.putText(frame, "SINALA | DACTILOLOGIA", (25, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, status_letter, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)

        # Texto transcrito geral
        transcribed = self.word_builder.full_text or "(nenhum texto)"
        cv2.putText(frame, f"TEXTO: {transcribed}", (25, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (56, 189, 248), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "[ESPACO] Separar palavra  |  [BACKSPACE] Apagar  |  [C] Limpar  |  [Q] Sair",
            (25, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (203, 213, 225),
            1,
            cv2.LINE_AA,
        )
