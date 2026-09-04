import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sinala.model.gru_sign_classifier import GRUSignClassifier

MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {requested}")


def set_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate_model(
    model: GRUSignClassifier,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    number_of_classes: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_items = 0
    total_correct = 0
    confusion = np.zeros((number_of_classes, number_of_classes), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for sequences, labels in loader:
            logits = model(sequences.to(device))
            loss = criterion(logits, labels.to(device))
            predictions = logits.argmax(dim=1).cpu()
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)
            total_correct += int((predictions == labels).sum().item())
            for target, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
                confusion[target, prediction] += 1
    per_class_f1: list[float] = []
    for class_index in range(number_of_classes):
        true_positive = confusion[class_index, class_index]
        false_positive = confusion[:, class_index].sum() - true_positive
        false_negative = confusion[class_index, :].sum() - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        per_class_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    accuracy = total_correct / total_items if total_items else 0.0
    return {
        "loss": total_loss / total_items if total_items else 0.0,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "confusion_matrix": confusion.tolist(),
    }


def train_model(
    model: GRUSignClassifier,
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    validation_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 42,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
) -> tuple[GRUSignClassifier, list[dict[str, float]]]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be finite and between 0 and 1")
    set_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_score = float("-inf")
    history: list[dict[str, float]] = []
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_items = 0
        for sequences, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences.to(device)), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)
        validation_metrics = evaluate_model(model, validation_dataset, device, model.number_of_classes, batch_size)
        train_loss = total_loss / total_items if total_items else 0.0
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "validation_macro_f1": validation_metrics["macro_f1"]})
        if validation_metrics["macro_f1"] > best_score:
            best_score = validation_metrics["macro_f1"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        logger.info("epoch={} train_loss={:.4f} validation_macro_f1={:.4f}", epoch + 1, train_loss, validation_metrics["macro_f1"])

    model.load_state_dict(best_state)
    return model, history


def save_checkpoint(path: Path, model: GRUSignClassifier, class_to_index: dict[str, int], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "number_of_classes": model.number_of_classes,
            "dropout": model.dropout,
        },
        "class_to_index": class_to_index,
        "index_to_class": {str(index): label for label, index in class_to_index.items()},
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[GRUSignClassifier, dict[str, Any]]:
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint file exceeds the supported size limit")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint model_config is missing or invalid")
    required_config = ("input_size", "hidden_size", "number_of_classes")
    if any(type(model_config.get(key)) is not int for key in required_config):
        raise ValueError("Checkpoint model dimensions must be integers")
    if model_config["input_size"] != 128 or not 1 <= model_config["hidden_size"] <= 2048 or not 2 <= model_config["number_of_classes"] <= 1024:
        raise ValueError("Checkpoint model dimensions are outside supported bounds")
    dropout = model_config.get("dropout", 0.2)
    if not isinstance(dropout, (int, float)) or not math.isfinite(float(dropout)) or not 0.0 <= dropout < 1.0:
        raise ValueError("Checkpoint dropout is invalid")
    model_config = {key: model_config[key] for key in required_config} | {"dropout": float(dropout)}
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("feature_size") != 128:
        raise ValueError("Checkpoint metadata is missing or invalid")
    index_to_class = payload.get("index_to_class")
    class_to_index = payload.get("class_to_index")
    expected_indices = {str(index) for index in range(model_config["number_of_classes"])}
    valid_class_to_index = (
        isinstance(class_to_index, dict)
        and len(class_to_index) == model_config["number_of_classes"]
        and all(isinstance(label, str) and type(index) is int for label, index in class_to_index.items())
    )
    if (
        not isinstance(index_to_class, dict)
        or set(index_to_class) != expected_indices
        or not all(isinstance(value, str) for value in index_to_class.values())
        or len(set(index_to_class.values())) != model_config["number_of_classes"]
        or not valid_class_to_index
        or {str(index): label for label, index in class_to_index.items()} != index_to_class
    ):
        raise ValueError("Checkpoint class mapping is invalid")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict) or not all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in state_dict.values()):
        raise ValueError("Checkpoint model_state_dict is missing or contains non-finite values")
    model = GRUSignClassifier(**model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, payload


__all__ = [
    "evaluate_model",
    "load_checkpoint",
    "resolve_device",
    "save_checkpoint",
    "set_seed",
    "train_model",
]
