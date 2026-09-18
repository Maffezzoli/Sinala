"""Unified real-time camera application fusing dynamic signs and fingerspelling."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from time import monotonic, perf_counter

import cv2
import numpy as np
import torch
from loguru import logger

from sinala.data.alphabet_dataset import normalize_hand_landmarks
from sinala.data.greetings_dataset import GREETINGS_CLASSES
from sinala.data.hand_landmark_extractor import HandLandmarkExtractor
from sinala.model.alphabet_classifier import AlphabetClassifier, load_alphabet_checkpoint
from sinala.model.gru_sign_classifier import GRUSignClassifier
from sinala.model.training import load_checkpoint
from sinala.runtime.word_builder import WordBuilder


class UnifiedCameraApp:
    """Unified camera transcriber for both whole words and manual alphabet."""

    def __init__(
        self,
        greetings_checkpoint_path: Path | str,
        alphabet_checkpoint_path: Path | str,
        model_asset_path: Path | str,
        camera_index: int = 0,
        alphabet_threshold: float = 0.70,
        greetings_threshold: float = 0.65,
        stability_frames: int = 12,
        sequence_length: int = 32,
        device: torch.device | None = None,
    ) -> None:
        self.greetings_checkpoint_path = Path(greetings_checkpoint_path)
        self.alphabet_checkpoint_path = Path(alphabet_checkpoint_path)
        self.model_asset_path = Path(model_asset_path)
        self.camera_index = camera_index
        self.alphabet_threshold = alphabet_threshold
        self.greetings_threshold = greetings_threshold
        self.greetings_margin_threshold = 0.20
        self.sequence_length = sequence_length
        self.device = device or torch.device("cpu")

        logger.info("Carregando modelo de dactilologia (alfabeto A-Z)...")
        self.alphabet_model: AlphabetClassifier = load_alphabet_checkpoint(self.alphabet_checkpoint_path, device=str(self.device))

        logger.info("Carregando modelo temporal de saudações e palavras...")
        self.greetings_model: GRUSignClassifier
        self.greetings_model, self.greetings_payload = load_checkpoint(self.greetings_checkpoint_path, self.device)
        metadata = self.greetings_payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("sequence_length") != self.sequence_length or metadata.get("feature_size") != 128:
            raise ValueError("Checkpoint de saudações incompatível com a janela temporal ou features")
        self.greetings_classes: dict[str, str] = self.greetings_payload["index_to_class"]
        if set(self.greetings_classes.values()) != set(GREETINGS_CLASSES):
            raise ValueError("Checkpoint de saudações não corresponde ao vocabulário conversacional")
        self.extractor = HandLandmarkExtractor(self.model_asset_path)
        self.word_builder = WordBuilder(
            stability_frames=stability_frames,
            confidence_threshold=alphabet_threshold,
        )
        # Buffers temporais para arbitragem
        self.frames_128: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.wrist_history: deque[tuple[float, float]] = deque(maxlen=15)
        self.last_word_commit_time: float = 0.0
        self.greeting_candidate: str | None = None
        self.greeting_candidate_count: int = 0
        self.greeting_armed: bool = True
        self.primary_handedness: str | None = None

    def run(self) -> None:
        """Run the unified camera loop."""
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera índice {self.camera_index}")

        logger.info("Câmera unificada do SINALA iniciada com sucesso. Pressione 'q' para sair.")
        previous_time = perf_counter()

        status_type = "AGUARDANDO"
        status_text = "AGUARDANDO SINAL"
        status_conf = 0.0

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Falha ao capturar quadro da webcam.")
                features, results = self.extractor.process(frame)

                now = monotonic()
                self.frames_128.append(features)

                displacement = 0.0
                detected_word: str | None = None

                if results.hand_landmarks:
                    hand_index = 0
                    if self.primary_handedness and results.handedness:
                        for index, candidates in enumerate(results.handedness):
                            if candidates and candidates[0].category_name.casefold() == self.primary_handedness:
                                hand_index = index
                                break
                    primary_lm = results.hand_landmarks[hand_index]
                    if results.handedness and hand_index < len(results.handedness) and results.handedness[hand_index]:
                        self.primary_handedness = results.handedness[hand_index][0].category_name.casefold()
                    self.wrist_history.append((primary_lm[0].x, primary_lm[0].y))

                    if len(self.wrist_history) >= 5:
                        xs = [p[0] for p in self.wrist_history]
                        ys = [p[1] for p in self.wrist_history]
                        displacement = float((max(xs) - min(xs)) + (max(ys) - min(ys)))
                    if displacement <= 0.04:
                        self.greeting_candidate = None
                        self.greeting_candidate_count = 0
                        self.greeting_armed = True

                    # 2. Avaliação Temporal de Sinais de Palavra (Greetings) pela GRU
                    if (
                        not detected_word
                        and self.greeting_armed
                        and len(self.frames_128) == self.sequence_length
                        and displacement > 0.04
                        and (now - self.last_word_commit_time) > 1.3
                    ):
                        word_candidate, w_conf, w_margin = self._predict_greeting()
                        if w_conf >= self.greetings_threshold and w_margin >= self.greetings_margin_threshold:
                            if word_candidate == self.greeting_candidate:
                                self.greeting_candidate_count += 1
                            else:
                                self.greeting_candidate = word_candidate
                                self.greeting_candidate_count = 1
                            status_type = "PALAVRA"
                            status_text = f"SINAL: {word_candidate}"
                            status_conf = w_conf
                            if self.greeting_candidate_count >= 3:
                                detected_word = word_candidate
                                self.word_builder.add_word(detected_word)
                                self.last_word_commit_time = now
                                self.frames_128.clear()
                                self.greeting_armed = False
                                self.greeting_candidate = None
                                self.greeting_candidate_count = 0
                                logger.info(f"Sinal reconhecido: {detected_word} ({w_conf * 100:.0f}%)")
                        else:
                            self.greeting_candidate = None
                            self.greeting_candidate_count = 0

                    # 2. Avaliação Espacial de Dactilologia (Alfabeto A-Z)
                    # Só executa quando a mão está estacionária e sem bloqueio recente de palavra
                    if not detected_word and (now - self.last_word_commit_time) > 0.8:
                        if displacement < 0.04:
                            is_left_hand = False
                            if results.handedness and hand_index < len(results.handedness) and results.handedness[hand_index]:
                                is_left_hand = results.handedness[hand_index][0].category_name.casefold() == "left"

                            pts = np.array([[lm.x, lm.y, lm.z] for lm in primary_lm], dtype=np.float32)
                            norm_pts = normalize_hand_landmarks(pts, is_left_hand=is_left_hand)
                            tensor_in = torch.from_numpy(norm_pts).to(self.device)
                            letter, l_conf = self.alphabet_model.predict_letter(tensor_in, threshold=self.alphabet_threshold)
                            self.word_builder.update(letter, l_conf)

                            if letter != "?":
                                status_type = "LETRA"
                                status_text = f"LETRA: {letter}"
                                status_conf = l_conf
                        else:
                            # Mão em movimento rápido sem sinal de palavra fechado
                            self.word_builder.update("?", 0.0)
                else:
                    self.primary_handedness = None
                    self.greeting_armed = True
                    self.wrist_history.clear()
                    self.frames_128.clear()
                    self.greeting_candidate = None
                    self.greeting_candidate_count = 0
                    self.word_builder.update("?", 0.0)
                    if (now - self.last_word_commit_time) > 1.0:
                        status_type = "AGUARDANDO"
                        status_text = "AGUARDANDO SINAL"
                        status_conf = 0.0

                self.extractor.draw_results(frame, results)

                current_time = perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time

                display = cv2.flip(frame, 1)
                self._draw_hud(display, status_type, status_text, status_conf, fps)

                cv2.imshow("SINALA - Reconhecimento Unificado (Palavras + Alfabeto)", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == 32:
                    self.word_builder.add_space()
                elif key in (8, 127):
                    self.word_builder.backspace()
                elif key == ord("c"):
                    self.word_builder.clear()
        finally:
            capture.release()
            self.extractor.close()
            cv2.destroyAllWindows()

    def _predict_greeting(self) -> tuple[str, float, float]:
        """Classify the current 32-frame sequence and return confidence margin."""
        seq_array = np.array(self.frames_128, dtype=np.float32)
        tensor_in = torch.from_numpy(seq_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.greetings_model(tensor_in)
            probs = torch.softmax(out, dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, k=2)
            confidence = float(top_probs[0].item())
            margin = float((top_probs[0] - top_probs[1]).item())
            pred_class = self.greetings_classes[str(top_indices[0].item())]
            return pred_class, confidence, margin

    def _draw_hud(self, frame: np.ndarray, status_type: str, status_text: str, conf: float, fps: float) -> None:
        """Render HUD with current detection and the accumulated sentence."""
        h, w, _ = frame.shape

        # Faixas superior e inferior escuras
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (10, 25, 47), -1)
        cv2.rectangle(overlay, (0, h - 80), (w, h), (10, 25, 47), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Barra de status superior
        cv2.putText(frame, "SINALA | SISTEMA UNIFICADO (PALAVRAS + ALFABETO)", (25, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 180, 255), 2, cv2.LINE_AA)

        if status_type == "PALAVRA":
            tag_color = (74, 222, 128)  # Verde claro para palavras completas
            display_top = f"{status_text} ({conf * 100:.0f}%) [SINAL RECONHECIDO]"
        elif status_type == "LETRA":
            tag_color = (255, 255, 255)
            progress = self.word_builder.get_progress()
            bars = int(progress * 10)
            prog_str = "[" + "=" * bars + " " * (10 - bars) + "]"
            display_top = f"{status_text} ({conf * 100:.0f}%) {prog_str}"
        else:
            tag_color = (148, 163, 184)
            display_top = status_text

        cv2.putText(frame, display_top, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.85, tag_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)

        # Barra inferior com texto acumulado
        full_text = self.word_builder.full_text or "(aguardando gestos...)"
        cv2.putText(frame, f"TEXTO: {full_text}", (25, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (56, 189, 248), 2, cv2.LINE_AA)
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
