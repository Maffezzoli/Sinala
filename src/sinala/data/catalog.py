import re
import unicodedata
from pathlib import Path

from loguru import logger

from sinala.data.dataset_sample import DatasetSample

OFFICIAL_CLASSES = (
    "Acontecer",
    "Aluno",
    "Amarelo",
    "América",
    "Aproveitar",
    "Bala",
    "Banco",
    "Banheiro",
    "Barulho",
    "Cinco",
    "Conhecer",
    "Espelho",
    "Esquina",
    "Filho",
    "Maçã",
    "Medo",
    "Ruim",
    "Sapo",
    "Vacina",
    "Vontade",
)


def canonicalize_label(value: str) -> str:
    normalized = _normalize(value)
    for label in OFFICIAL_CLASSES:
        if _normalize(label) == normalized:
            return label
    raise ValueError(f"Unknown dataset class: {value}")


_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
_SIGNER_PATTERN = re.compile(r"(?:sinalizador|signer)[_ -]?(\d+)", re.IGNORECASE)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if not unicodedata.combining(character)).casefold()


def _find_label(path: Path) -> str | None:
    known_labels = {_normalize(label): label for label in OFFICIAL_CLASSES}
    for part in (path.parent.name, path.stem, *path.parts):
        normalized = _normalize(part)
        for normalized_label, label in known_labels.items():
            if normalized == normalized_label or normalized_label in normalized:
                return label
    return None


def _find_signer_id(path: Path) -> str | None:
    match = _SIGNER_PATTERN.search(str(path))
    return match.group(1).zfill(2) if match else None


def catalog_dataset(data_root: Path, selected_classes: set[str] | None = None) -> list[DatasetSample]:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")

    allowed = {_normalize(value) for value in selected_classes} if selected_classes else None
    samples: list[DatasetSample] = []
    invalid_paths: list[Path] = []

    for video_path in sorted(
        path for path in data_root.rglob("*") if path.suffix.casefold() in _VIDEO_EXTENSIONS and not path.name.startswith("._") and "__MACOSX" not in path.parts
    ):
        label = _find_label(video_path)
        signer_id = _find_signer_id(video_path)
        if label is None or signer_id is None:
            invalid_paths.append(video_path)
            continue
        if allowed is not None and _normalize(label) not in allowed:
            continue
        samples.append(DatasetSample(video_path=video_path, label=label, signer_id=signer_id))

    if invalid_paths:
        logger.warning("{} videos could not be cataloged", len(invalid_paths))
    return samples


def write_manifest(samples: list[DatasetSample], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(sample.model_dump_json() for sample in samples) + "\n", encoding="utf-8")


def read_manifest(manifest_path: Path) -> list[DatasetSample]:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    return [DatasetSample.model_validate_json(line) for line in lines if line.strip()]


def summarize_samples(samples: list[DatasetSample]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for sample in samples:
        summary.setdefault(sample.label, {}).setdefault(sample.signer_id, 0)
        summary[sample.label][sample.signer_id] += 1
    return summary


__all__ = ["OFFICIAL_CLASSES", "canonicalize_label", "catalog_dataset", "read_manifest", "summarize_samples", "write_manifest"]
