"""Unit tests for Libras fingerspelling alphabet dataset, classifier, and word builder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sinala.cli import _build_parser
from sinala.data.alphabet_dataset import ALPHABET_CLASSES, normalize_hand_landmarks
from sinala.model.alphabet_classifier import (
    AlphabetClassifier,
    load_alphabet_checkpoint,
    save_alphabet_checkpoint,
)
from sinala.runtime.word_builder import WordBuilder


def test_alphabet_classes_count() -> None:
    assert len(ALPHABET_CLASSES) == 26
    assert ALPHABET_CLASSES[0] == "A"
    assert ALPHABET_CLASSES[-1] == "Z"


def test_normalize_hand_landmarks_shape_and_invariance() -> None:
    rng = np.random.default_rng(42)
    raw_coords = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)

    wrist_shift = np.array([10.0, -5.0, 3.0], dtype=np.float32)
    shifted_coords = raw_coords + wrist_shift

    norm_orig = normalize_hand_landmarks(raw_coords, is_left_hand=False)
    norm_shifted = normalize_hand_landmarks(shifted_coords, is_left_hand=False)

    assert norm_orig.shape == (225,)
    assert norm_shifted.shape == (225,)
    np.testing.assert_allclose(norm_orig, norm_shifted, atol=1e-5)


def test_normalize_hand_landmarks_handedness_mirroring() -> None:
    rng = np.random.default_rng(123)
    coords = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)

    norm_right = normalize_hand_landmarks(coords, is_left_hand=False)
    norm_left = normalize_hand_landmarks(coords, is_left_hand=True)

    # Distâncias e ângulos não dependem da lateralidade ou reflexão horizontal.
    np.testing.assert_allclose(norm_right, norm_left, atol=1e-5)


def test_alphabet_classifier_forward() -> None:
    model = AlphabetClassifier(in_features=225, num_classes=26)
    model.eval()

    batch_in = torch.randn(4, 225)
    out = model(batch_in)
    assert out.shape == (4, 26)

    single_in = torch.randn(225)
    single_out = model(single_in)
    assert single_out.shape == (26,)

    letter, conf = model.predict_letter(single_in, threshold=0.0, minimum_margin=0.0)
    assert letter in ALPHABET_CLASSES
    assert 0.0 <= conf <= 1.0
    unknown_letter, _ = model.predict_letter(single_in, threshold=0.0, minimum_margin=2.0)
    assert unknown_letter == "?"


def test_alphabet_checkpoint_roundtrip(tmp_path) -> None:
    model = AlphabetClassifier(in_features=225, num_classes=26)
    ckpt_file = tmp_path / "test_alphabet.pt"

    save_alphabet_checkpoint(model, ckpt_file, metadata={"test": True})
    assert ckpt_file.exists()

    loaded_model = load_alphabet_checkpoint(ckpt_file, device="cpu")
    assert isinstance(loaded_model, AlphabetClassifier)
    assert loaded_model.in_features == 225
    assert loaded_model.num_classes == 26


def test_alphabet_checkpoint_rejects_incompatible_schema(tmp_path) -> None:
    checkpoint = tmp_path / "wrong_schema.pt"
    model = AlphabetClassifier(in_features=63, num_classes=26)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_features": 63,
            "num_classes": 26,
            "classes": list(ALPHABET_CLASSES),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="dimensão"):
        load_alphabet_checkpoint(checkpoint)


def test_word_builder_stability_and_debounce() -> None:
    builder = WordBuilder(stability_frames=3, confidence_threshold=0.75, repeat_cooldown_seconds=1.0)

    # Frame 1: confiança baixa -> descartado
    assert builder.update("A", 0.50) is None
    assert builder.current_word == ""

    # Frame 2: 'A' com confiança alta (contagem = 1)
    assert builder.update("A", 0.85) is None
    assert builder.current_word == ""

    # Frame 3: 'A' com confiança alta (contagem = 2)
    assert builder.update("A", 0.85) is None
    assert builder.current_word == ""

    # Frame 4: 'A' com confiança alta (contagem = 3 -> estabilizou!)
    committed = builder.update("A", 0.85)
    assert committed == "A"
    assert builder.current_word == "A"

    # Frame 5: Mantém 'A' imediatamente -> debounce não duplica 'A'
    assert builder.update("A", 0.85) is None
    assert builder.current_word == "A"
    for _ in range(6):
        assert builder.update("A", 0.85) is None
    assert builder.full_text == "A"

    # Predições rejeitadas rearmam a letra e não acumulam estabilidade.
    assert builder.update("?", 0.0) is None
    builder.update("C", 0.90)
    assert builder.update("?", 0.0) is None
    builder.update("C", 0.90)
    assert builder.current_word == "A"

    # Agora sinaliza 'B' por 3 frames
    builder.update("B", 0.90)
    builder.update("B", 0.90)
    committed_b = builder.update("B", 0.90)
    assert committed_b == "B"
    assert builder.current_word == "AB"

    # Testa espaço e nova palavra
    builder.add_space()
    assert builder.current_word == ""
    assert builder.full_text == "AB"

    builder.update("C", 0.90)
    builder.update("C", 0.90)
    builder.update("C", 0.90)
    assert builder.full_text == "AB C"

    # Testa backspace
    builder.backspace()
    assert builder.full_text == "AB"

    # Testa clear
    builder.clear()

    # Testa add_word diretamente
    builder.add_word("OI")
    assert builder.full_text == "OI"
    builder.update("D", 0.90)
    builder.update("D", 0.90)
    builder.update("D", 0.90)
    assert builder.full_text == "OI D"


def test_cli_alphabet_subparsers() -> None:
    parser = _build_parser()

    # Testa parsing de download-alphabet
    args_dl = parser.parse_args(["download-alphabet", "--output", "custom/dir"])
    assert args_dl.command == "download-alphabet"
    assert args_dl.output == "custom/dir"

    # Testa parsing de train-alphabet
    args_tr = parser.parse_args(["train-alphabet", "--epochs", "5", "--batch-size", "32"])
    assert args_tr.command == "train-alphabet"
    assert args_tr.epochs == 5
    assert args_tr.batch_size == 32

    # Testa parsing de camera-alphabet
    args_cam = parser.parse_args(["camera-alphabet", "--confidence-threshold", "0.80"])
    assert args_cam.command == "camera-alphabet"
    assert args_cam.confidence_threshold == 0.80

    # Testa parsing de camera-unified
    args_uni = parser.parse_args(["camera-unified", "--greetings-threshold", "0.75"])
    assert args_uni.command == "camera-unified"
    assert args_uni.greetings_threshold == 0.75
