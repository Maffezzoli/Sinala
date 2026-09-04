from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DatasetSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_path: Path
    label: str = Field(min_length=1)
    signer_id: str = Field(min_length=1)
    frame_count: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, gt=0.0)
