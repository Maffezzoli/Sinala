"""Unit tests for greetings dataset and CLI command."""

from __future__ import annotations

from sinala.cli import _build_parser
from sinala.data.greetings_dataset import GREETINGS_CLASSES


def test_greetings_classes() -> None:
    assert len(GREETINGS_CLASSES) == 9
    assert "Oi" in GREETINGS_CLASSES
    assert "Nome" in GREETINGS_CLASSES
    assert "Obrigado" in GREETINGS_CLASSES
    assert "Por favor" in GREETINGS_CLASSES


def test_cli_download_greetings_subparser() -> None:
    parser = _build_parser()
    args = parser.parse_args(["download-greetings", "--output", "custom/greetings"])
    assert args.command == "download-greetings"
    assert args.output == "custom/greetings"
    assert args.sequence_length == 32


def test_cli_train_greetings_subparser() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train-greetings", "--epochs", "2", "--augment-copies", "3"])
    assert args.command == "train-greetings"
    assert args.epochs == 2
    assert args.augment_copies == 3
