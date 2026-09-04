from pathlib import Path

import pytest
import torch

from sinala.model.gru_sign_classifier import GRUSignClassifier
from sinala.model.training import load_checkpoint


def test_gru_sign_classifier_when_batch_is_valid_then_returns_logits_per_class() -> None:
    model = GRUSignClassifier(input_size=128, hidden_size=16, number_of_classes=3)

    logits = model(torch.zeros((4, 32, 128)))

    assert logits.shape == (4, 3)


def test_gru_sign_classifier_when_feature_dimension_is_invalid_then_rejects_input() -> None:
    model = GRUSignClassifier(input_size=128, hidden_size=16, number_of_classes=3)

    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.zeros((4, 32, 64)))


def test_gru_sign_classifier_when_dropout_is_out_of_range_then_rejects_configuration() -> None:
    with pytest.raises(ValueError, match="dropout"):
        GRUSignClassifier(input_size=128, hidden_size=16, number_of_classes=3, dropout=1.0)


def test_load_checkpoint_when_dimensions_exceed_bounds_then_rejects_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_config": {"input_size": 128, "hidden_size": 10_000_000, "number_of_classes": 3},
            "metadata": {"feature_size": 128},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="outside supported bounds"):
        load_checkpoint(checkpoint_path, torch.device("cpu"))


def test_load_checkpoint_when_class_mappings_disagree_then_rejects_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_config": {"input_size": 128, "hidden_size": 16, "number_of_classes": 2},
            "class_to_index": {"Acontecer": 1, "Aluno": 0},
            "index_to_class": {"0": "Acontecer", "1": "Aluno"},
            "metadata": {"feature_size": 128},
            "model_state_dict": {},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="class mapping"):
        load_checkpoint(checkpoint_path, torch.device("cpu"))


def test_load_checkpoint_when_weights_are_nonfinite_then_rejects_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_config": {"input_size": 128, "hidden_size": 16, "number_of_classes": 2},
            "class_to_index": {"Acontecer": 0, "Aluno": 1},
            "index_to_class": {"0": "Acontecer", "1": "Aluno"},
            "metadata": {"feature_size": 128},
            "model_state_dict": {"weight": torch.tensor(float("nan"))},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="non-finite"):
        load_checkpoint(checkpoint_path, torch.device("cpu"))
