"""Download and preprocessing of multi-source Libras conversational signs."""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger

from sinala.data.hand_landmark_extractor import HandLandmarkExtractor
from sinala.data.landmark_preprocessor import LandmarkPreprocessor

GREETINGS_CLASSES: tuple[str, ...] = (
    "Ajudar",
    "Cumprimento",
    "Desculpa",
    "Não",
    "Nome",
    "Obrigado",
    "Oi",
    "Por favor",
    "Sim",
)

ANNOTATIONS_URL = "https://huggingface.co/datasets/ibmectech/v-librasil-raw/raw/main/annotations.csv"
VIDEO_BASE_URL = "https://huggingface.co/datasets/ibmectech/v-librasil-raw/resolve/main/videos/"

EXTERNAL_GREETING_SOURCES: tuple[dict[str, str], ...] = (
    {
        "source_id": "acessibilidade_brasil",
        "label": "Oi",
        "signer_id": "source_oi",
        "variant": "oi",
        "url": "http://www.acessibilidadebrasil.org.br/libras_3/public/media/palavras/videos/oiSm_Prog001.mp4",
    },
    {
        "source_id": "acessibilidade_brasil",
        "label": "Nome",
        "signer_id": "source_nome",
        "variant": "nome",
        "url": "http://www.acessibilidadebrasil.org.br/libras_3/public/media/palavras/videos/nomeSm_Prog001.mp4",
    },
    {
        "source_id": "acessibilidade_brasil",
        "label": "Obrigado",
        "signer_id": "source_obrigado_1",
        "variant": "obrigado1",
        "url": "http://www.acessibilidadebrasil.org.br/libras_3/public/media/palavras/videos/obrigado1Sm_Prog001.mp4",
    },
    {
        "source_id": "acessibilidade_brasil",
        "label": "Obrigado",
        "signer_id": "source_obrigado_2",
        "variant": "obrigado2",
        "url": "http://www.acessibilidadebrasil.org.br/libras_3/public/media/palavras/videos/obrigado2Sm_Prog001.mp4",
    },
    {
        "source_id": "spread_the_sign",
        "label": "Oi",
        "signer_id": "source_ola",
        "variant": "ola",
        "url": "https://media.spreadthesign.com/video/mp4/14/555228.mp4",
    },
)


def _v_librasil_sources() -> list[dict[str, str]]:
    request = urllib.request.Request(ANNOTATIONS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        reader = csv.DictReader(response.read().decode("utf-8").splitlines())
        selected: list[dict[str, str]] = []
        for row in reader:
            label = row["class"].strip()
            if label not in GREETINGS_CLASSES:
                continue
            video_name = row["video_name"].strip()
            selected.append(
                {
                    "source_id": "v_librasil",
                    "label": label,
                    "signer_id": row["user_id"].strip(),
                    "variant": Path(video_name).stem,
                    "video_name": video_name,
                    "url": f"{VIDEO_BASE_URL}{urllib.parse.quote(video_name)}",
                }
            )
    return selected


def _source_filename(source: dict[str, str]) -> str:
    return "__".join((source["source_id"], source["label"], source["signer_id"], source["variant"])) + ".mp4"


def download_greetings_dataset(
    target_video_dir: str | Path = "data/raw/greetings/videos",
    target_landmarks_dir: str | Path = "data/landmarks-greetings",
    model_asset_path: str | Path = "models/hand_landmarker.task",
    sequence_length: int = 32,
) -> tuple[Path, Path]:
    """Download and extract conversational signs from V-Librasil and curated public sources."""
    video_dir = Path(target_video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir = Path(target_landmarks_dir)
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    sources = _v_librasil_sources() + [dict(source) for source in EXTERNAL_GREETING_SOURCES]
    logger.info(f"Fontes conversacionais selecionadas: {len(sources)} vídeos")

    def fetch(source: dict[str, str]) -> tuple[dict[str, str], int]:
        destination = video_dir / _source_filename(source)
        if destination.exists() and destination.stat().st_size > 10_000:
            return source, destination.stat().st_size
        request = urllib.request.Request(source["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            output.write(response.read())
        return source, destination.stat().st_size

    with ThreadPoolExecutor(max_workers=6) as executor:
        downloaded = list(executor.map(fetch, sources))
    logger.info(f"Download concluído: {len(downloaded)} vídeos ({sum(size for _, size in downloaded) / (1024 * 1024):.2f} MB)")

    extractor = HandLandmarkExtractor(Path(model_asset_path))
    preprocessor = LandmarkPreprocessor(extractor=extractor, sequence_length=sequence_length)
    manifest_path = landmarks_dir / "manifest.jsonl"
    csv_path = landmarks_dir / "greetings_landmarks.csv"
    records: list[dict[str, object]] = []
    csv_rows: list[list[object]] = []

    try:
        for source, _ in downloaded:
            video_path = video_dir / _source_filename(source)
            sequence = preprocessor.process_video(video_path)
            sequence_path = landmarks_dir / f"{video_path.stem}.npy"
            np.save(sequence_path, sequence)
            records.append(
                {
                    "video_path": str(sequence_path),
                    "label": source["label"],
                    "signer_id": source["signer_id"],
                    "source_id": source["source_id"],
                    "variant": source["variant"],
                    "frames": len(sequence),
                    "features": int(sequence.shape[1]),
                }
            )
            for frame_index, frame_features in enumerate(sequence):
                csv_rows.append([source["label"], source["signer_id"], source["source_id"], frame_index, *map(float, frame_features)])
    finally:
        extractor.close()

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["label", "signer_id", "source_id", "frame_idx"] + [f"feat_{i}" for i in range(128)])
        writer.writerows(csv_rows)

    logger.info(f"Landmarks conversacionais extraídos: {len(records)} sequências em {landmarks_dir}")
    return landmarks_dir, csv_path
