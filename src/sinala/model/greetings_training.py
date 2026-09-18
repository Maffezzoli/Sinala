"""Training utilities for the multi-source conversational-sign classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import TensorDataset

from sinala.data.landmark_augmenter import LandmarkAugmenter
from sinala.model.gru_sign_classifier import GRUSignClassifier
from sinala.model.training import evaluate_model, save_checkpoint, train_model


def train_greetings_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    epochs: int = 30,
    augment_copies: int = 40,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train greetings using source-aware validation and save the standard checkpoint.

    Articulador3 is held out for validation. All external sources augment the training
    split, so the model learns source variation while preserving an unseen V-Librasil
    articulator for model selection.
    """
    records = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = sorted({record["label"] for record in records})
    class_to_index = {label: index for index, label in enumerate(labels)}

    train_records = [record for record in records if record["signer_id"] != "Articulador3"]
    validation_records = [record for record in records if record["signer_id"] == "Articulador3"]
    if {record["label"] for record in train_records} != set(labels) or {record["label"] for record in validation_records} != set(labels):
        raise ValueError("Treino e validação precisam conter todas as classes conversacionais")

    augmenter = LandmarkAugmenter(probability=1.0, noise_std=0.015, scale_range=(0.85, 1.15), temporal_speed_range=(0.85, 1.15), seed=42)
    train_sequences: list[np.ndarray] = []
    train_labels: list[int] = []
    for record in train_records:
        sequence = np.load(record["video_path"]).astype(np.float32)
        train_sequences.append(sequence)
        train_labels.append(class_to_index[record["label"]])
        for _ in range(augment_copies):
            train_sequences.append(augmenter(sequence))
            train_labels.append(class_to_index[record["label"]])

    validation_sequences = [np.load(record["video_path"]).astype(np.float32) for record in validation_records]
    validation_labels = [class_to_index[record["label"]] for record in validation_records]
    train_dataset = TensorDataset(torch.from_numpy(np.asarray(train_sequences)), torch.tensor(train_labels, dtype=torch.long))
    validation_dataset = TensorDataset(torch.from_numpy(np.asarray(validation_sequences)), torch.tensor(validation_labels, dtype=torch.long))

    torch_device = device or torch.device("cpu")
    model = GRUSignClassifier(input_size=128, hidden_size=64, number_of_classes=len(labels), dropout=0.2)
    model, history = train_model(
        model,
        train_dataset,
        validation_dataset,
        torch_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        label_smoothing=0.05,
    )
    validation_metrics = evaluate_model(model, validation_dataset, torch_device, len(labels), batch_size)

    # Após medir generalização em Articulador3, retreina o artefato final com todas as fontes.
    full_augmenter = LandmarkAugmenter(
        probability=1.0,
        noise_std=0.015,
        scale_range=(0.85, 1.15),
        temporal_speed_range=(0.85, 1.15),
        seed=43,
    )
    full_sequences: list[np.ndarray] = []
    full_labels: list[int] = []
    for record in records:
        sequence = np.load(record["video_path"]).astype(np.float32)
        full_sequences.append(sequence)
        full_labels.append(class_to_index[record["label"]])
        for _ in range(augment_copies):
            full_sequences.append(full_augmenter(sequence))
            full_labels.append(class_to_index[record["label"]])
    full_dataset = TensorDataset(torch.from_numpy(np.asarray(full_sequences)), torch.tensor(full_labels, dtype=torch.long))
    final_model = GRUSignClassifier(input_size=128, hidden_size=64, number_of_classes=len(labels), dropout=0.2)
    final_model, _ = train_model(
        final_model,
        full_dataset,
        full_dataset,
        torch_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        label_smoothing=0.05,
    )

    metadata = {
        "feature_size": 128,
        "sequence_length": 32,
        "classes": labels,
        "training_sources": sorted({record["source_id"] for record in records}),
        "validation_signer": "Articulador3",
        "train_sequences": len(train_records),
        "augmented_train_sequences": len(train_dataset),
        "deployment_sequences": len(full_dataset),
        "validation_sequences": len(validation_dataset),
        "validation_macro_f1": validation_metrics["macro_f1"],
    }
    save_checkpoint(Path(checkpoint_path), final_model, class_to_index, metadata)
    logger.info("Checkpoint conversacional salvo em {} com macro-F1 holdout {:.4f}", checkpoint_path, validation_metrics["macro_f1"])
    return {"labels": labels, "history": history, "validation_metrics": validation_metrics, "metadata": metadata}
