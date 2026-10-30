"""Model export utilities to convert PyTorch checkpoints to ONNX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from sinala.model.alphabet_classifier import AlphabetClassifier, load_alphabet_checkpoint
from sinala.model.training import load_checkpoint


class OnnxExporter:
    """Exports trained Libras models to standardized ONNX runtime formats."""

    def __init__(self, output_dir: str | Path = "web/models") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_alphabet(
        self,
        checkpoint_path: str | Path,
        filename: str = "alphabet.onnx",
        opset_version: int = 17,
    ) -> Path:
        """Export the AlphabetClassifier MLP model to ONNX."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint do alfabeto nao encontrado: {ckpt_path}")

        model: AlphabetClassifier = load_alphabet_checkpoint(ckpt_path, device="cpu")
        model.eval()

        output_path = self.output_dir / filename
        dummy_input = torch.randn(1, model.in_features, dtype=torch.float32)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            dynamo=False,
        )

        metadata_path = self.output_dir / "alphabet_metadata.json"
        metadata: dict[str, Any] = {
            "in_features": model.in_features,
            "num_classes": model.num_classes,
            "classes": [chr(i) for i in range(ord("A"), ord("Z") + 1)],
            "model_type": "AlphabetClassifier",
            "format": "onnx",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"Modelo de alfabeto exportado com sucesso para {output_path}")
        return output_path

    def export_greetings(
        self,
        checkpoint_path: str | Path,
        filename: str = "greetings.onnx",
        opset_version: int = 17,
    ) -> Path:
        """Export the GreetingGRU model to ONNX."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint de saudacoes nao encontrado: {ckpt_path}")

        model, payload = load_checkpoint(ckpt_path, device=torch.device("cpu"))
        model.eval()

        output_path = self.output_dir / filename
        sequence_length = int(payload.get("metadata", {}).get("sequence_length", 32))
        input_size = int(model.input_size)
        dummy_input = torch.randn(1, sequence_length, input_size, dtype=torch.float32)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["sequence"],
            output_names=["logits"],
            dynamic_axes={
                "sequence": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            dynamo=False,
        )

        index_to_class: dict[str, str] = payload.get("index_to_class", {})
        classes = [index_to_class[str(i)] for i in range(len(index_to_class))]
        metadata_path = self.output_dir / "greetings_metadata.json"
        metadata: dict[str, Any] = {
            "input_size": input_size,
            "hidden_size": model.hidden_size,
            "sequence_length": sequence_length,
            "num_classes": model.number_of_classes,
            "classes": classes,
            "model_type": "GRUSignClassifier",
            "format": "onnx",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"Modelo de saudacoes exportado com sucesso para {output_path}")
        return output_path
