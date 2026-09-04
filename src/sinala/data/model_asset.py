from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

MODEL_ASSET_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def download_hand_model(destination: Path, timeout_seconds: float = 120.0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    with httpx.stream("GET", MODEL_ASSET_URL, follow_redirects=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                output.write(chunk)
    if temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("Downloaded MediaPipe model is empty")
    temporary_path.replace(destination)
    return destination


__all__ = ["MODEL_ASSET_URL", "download_hand_model"]
