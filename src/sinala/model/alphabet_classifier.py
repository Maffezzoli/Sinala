"""PyTorch classifier and training utilities for the 26-letter Libras Alphabet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from sinala.data.alphabet_dataset import ALPHABET_CLASSES, FEATURE_DIM_ALPHABET, LibrasAlphabetDataset


class AlphabetClassifier(nn.Module):
    """Multilayer Perceptron for real-time Libras fingerspelling recognition."""

    def __init__(self, in_features: int = 225, num_classes: int = len(ALPHABET_CLASSES), dropout: float = 0.25) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects shape (B, 225) or (225,)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            out = self.net(x)
            return out.squeeze(0)
        return self.net(x)

    @torch.no_grad()
    def predict_letter(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        minimum_margin: float = 0.15,
    ) -> tuple[str, float]:
        """Predict a letter only when confidence and top-two margin are sufficient."""
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, k=2)
        confidence = float(top_probs[0].item())
        margin = float((top_probs[0] - top_probs[1]).item())
        if confidence < threshold or margin < minimum_margin:
            return "?", confidence
        return ALPHABET_CLASSES[int(top_indices[0].item())], confidence


def train_alphabet_model(
    train_dataset: LibrasAlphabetDataset,
    val_dataset: LibrasAlphabetDataset,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> tuple[AlphabetClassifier, dict[str, Any]]:
    """Train the alphabet classifier using PyTorch AdamW."""
    torch_device = torch.device(device)
    model = AlphabetClassifier(
        in_features=train_dataset.features.shape[1],
        num_classes=len(ALPHABET_CLASSES),
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    history: dict[str, list[float]] = {"train_loss": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(torch_device), by.to(torch_device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        train_loss = total_loss / len(train_dataset)
        val_acc = evaluate_alphabet_model(model, val_loader, torch_device)

        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        logger.info(f"Época {epoch:02d}/{epochs} | Loss Treino: {train_loss:.4f} | Acurácia Validação: {val_acc * 100:.2f}%")

    return model, history


def evaluate_alphabet_model(model: AlphabetClassifier, data_loader: DataLoader, device: torch.device) -> float:
    """Evaluate accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in data_loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx).argmax(dim=-1)
            correct += int((preds == by).sum().item())
            total += len(by)
    return (correct / total) if total > 0 else 0.0


def save_alphabet_checkpoint(model: AlphabetClassifier, checkpoint_path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """Save alphabet model weights and metadata."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "in_features": model.in_features,
        "num_classes": model.num_classes,
        "classes": list(ALPHABET_CLASSES),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    logger.info(f"Checkpoint do alfabeto salvo em {path}")
    return path


def load_alphabet_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> AlphabetClassifier:
    """Load alphabet model weights with weights_only=True security."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint do alfabeto deve ser um dicionário")
    if checkpoint.get("in_features") != FEATURE_DIM_ALPHABET:
        raise ValueError("Checkpoint do alfabeto usa dimensão de features incompatível")
    if checkpoint.get("num_classes") != len(ALPHABET_CLASSES):
        raise ValueError("Checkpoint do alfabeto usa quantidade de classes incompatível")
    if checkpoint.get("classes") != list(ALPHABET_CLASSES):
        raise ValueError("Checkpoint do alfabeto usa ordem de classes incompatível")
    model = AlphabetClassifier(
        in_features=FEATURE_DIM_ALPHABET,
        num_classes=len(ALPHABET_CLASSES),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model
