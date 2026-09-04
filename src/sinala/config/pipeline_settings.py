from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SINALA_", env_file=".env", extra="ignore")

    data_root: Path = Path("data/raw")
    landmarks_root: Path = Path("data/landmarks")
    checkpoints_root: Path = Path("artifacts")
    selected_classes: list[str] = Field(default_factory=list)
    selected_signers: list[str] = Field(default_factory=list)
    sequence_length: int = Field(default=32, ge=2, le=256)
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    stability_window_size: int = Field(default=5, ge=1, le=30)
    minimum_consensus: int = Field(default=4, ge=1, le=30)
    cooldown_seconds: float = Field(default=0.8, ge=0.0, le=30.0)
    camera_index: int = Field(default=0, ge=0)
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    epochs: int = Field(default=30, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=512)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.0001, ge=0.0, le=1.0)
    label_smoothing: float = Field(default=0.05, ge=0.0, lt=1.0)
    augmentation_probability: float = Field(default=0.8, ge=0.0, le=1.0)
    augmentation_noise_std: float = Field(default=0.01, ge=0.0, le=1.0)
    hidden_size: int = Field(default=128, ge=1, le=2048)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    device: Literal["auto", "cpu", "mps"] = "auto"

    @field_validator("selected_classes", "selected_signers")
    @classmethod
    def normalize_values(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @field_validator("minimum_consensus")
    @classmethod
    def consensus_cannot_exceed_window(cls, value: int, info: object) -> int:
        window_size = info.data.get("stability_window_size", 5)
        if value > window_size:
            raise ValueError("minimum_consensus cannot exceed stability_window_size")
        return value
