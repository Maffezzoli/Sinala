import torch
from torch import nn


class GRUSignClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, number_of_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or number_of_classes < 2:
            raise ValueError("input_size, hidden_size and number_of_classes must be positive; at least two classes are required")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.number_of_classes = number_of_classes
        self.dropout = dropout
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, number_of_classes)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 3 or sequences.shape[-1] != self.input_size:
            raise ValueError(f"Expected input shape [batch, time, {self.input_size}], got {tuple(sequences.shape)}")
        _, hidden = self.gru(sequences)
        return self.classifier(self.dropout_layer(hidden[-1]))


__all__ = ["GRUSignClassifier"]
