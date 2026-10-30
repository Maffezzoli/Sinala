"""Unit tests for the OnnxExporter model utility."""

from pathlib import Path

import pytest

from sinala.data.alphabet_dataset import ALPHABET_CLASSES, FEATURE_DIM_ALPHABET
from sinala.model.alphabet_classifier import AlphabetClassifier, save_alphabet_checkpoint
from sinala.model.gru_sign_classifier import GRUSignClassifier
from sinala.model.onnx_exporter import OnnxExporter
from sinala.model.training import save_checkpoint


def test_onnx_exporter_when_alphabet_model_then_exports_valid_onnx(tmp_path: Path) -> None:
    # Arrange dummy checkpoint
    ckpt_path = tmp_path / "alphabet.pt"
    model = AlphabetClassifier(in_features=FEATURE_DIM_ALPHABET, num_classes=len(ALPHABET_CLASSES))
    save_alphabet_checkpoint(model, ckpt_path)

    output_dir = tmp_path / "web_models"
    exporter = OnnxExporter(output_dir=output_dir)

    # Act
    exported_file = exporter.export_alphabet(ckpt_path, filename="alphabet.onnx")

    # Assert
    assert exported_file.exists()
    assert exported_file.stat().st_size > 0
    assert (output_dir / "alphabet_metadata.json").exists()


def test_onnx_exporter_when_greetings_model_then_exports_valid_onnx(tmp_path: Path) -> None:
    # Arrange dummy greetings checkpoint
    ckpt_path = tmp_path / "greetings.pt"
    classes = {"Oi": 0, "Obrigado": 1, "Sim": 2}
    model = GRUSignClassifier(input_size=128, hidden_size=64, number_of_classes=3)
    save_checkpoint(
        ckpt_path,
        model,
        class_to_index=classes,
        metadata={"feature_size": 128, "sequence_length": 32, "classes": list(classes.keys())},
    )

    output_dir = tmp_path / "web_models"
    exporter = OnnxExporter(output_dir=output_dir)

    # Act
    exported_file = exporter.export_greetings(ckpt_path, filename="greetings.onnx")

    # Assert
    assert exported_file.exists()
    assert exported_file.stat().st_size > 0
    assert (output_dir / "greetings_metadata.json").exists()


def test_onnx_exporter_when_missing_checkpoint_then_raises_file_not_found(tmp_path: Path) -> None:
    exporter = OnnxExporter(output_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="nao encontrado"):
        exporter.export_alphabet(tmp_path / "non_existent.pt")

    with pytest.raises(FileNotFoundError, match="nao encontrado"):
        exporter.export_greetings(tmp_path / "non_existent_greetings.pt")
