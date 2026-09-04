import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from loguru import logger

from sinala.config.pipeline_settings import PipelineSettings
from sinala.data.catalog import canonicalize_label, catalog_dataset, summarize_samples
from sinala.data.dataset_sample import DatasetSample
from sinala.data.hand_landmark_extractor import HandLandmarkExtractor
from sinala.data.landmark_augmenter import LandmarkAugmenter
from sinala.data.landmark_dataset import LandmarkSequenceDataset
from sinala.data.landmark_preprocessor import LandmarkPreprocessor
from sinala.data.minds_downloader import MindsDatasetDownloader
from sinala.data.model_asset import download_hand_model
from sinala.data.splits import build_signer_split
from sinala.model.gru_sign_classifier import GRUSignClassifier
from sinala.model.training import evaluate_model, resolve_device, save_checkpoint, set_seed, train_model
from sinala.runtime.realtime_caption_app import RealtimeCaptionApp


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _signer_set(value: str) -> set[str]:
    return {item.zfill(2) for item in _csv_values(value)}


def _inspect_data(args: argparse.Namespace) -> None:
    samples = catalog_dataset(Path(args.data_root), set(_csv_values(args.classes)) or None)
    summary = summarize_samples(samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _download_data(args: argparse.Namespace) -> None:
    signers = [item.zfill(2) for item in _csv_values(args.signers)]
    paths = MindsDatasetDownloader(Path(args.output)).download_signers(signers)
    for path in paths:
        print(path)


def _download_model(args: argparse.Namespace) -> None:
    print(download_hand_model(Path(args.output)))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _landmark_cache_key(video_path: Path, model_asset_path: Path, sequence_length: int, model_digest: str | None = None) -> str:
    source = ":".join(
        (
            "landmarks-v2",
            str(video_path.resolve()),
            _file_digest(video_path),
            str(model_asset_path.resolve()),
            model_digest or _file_digest(model_asset_path),
            str(sequence_length),
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]


def _extract_landmarks(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_root = Path(args.output)
    samples = catalog_dataset(data_root, set(_csv_values(args.classes)) or None)
    if not samples:
        raise ValueError("No videos found for landmark extraction")
    model_asset_path = Path(args.model_asset)
    model_digest = _file_digest(model_asset_path)
    extractor = HandLandmarkExtractor(model_asset_path)
    preprocessor = LandmarkPreprocessor(extractor, args.sequence_length)
    manifest_path = output_root / "manifest.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    try:
        for sample in samples:
            sample_id = _landmark_cache_key(sample.video_path, model_asset_path, args.sequence_length, model_digest)
            sequence_path = output_root / f"{sample_id}.npy"
            valid_cache = False
            if sequence_path.exists():
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        cached_sequence = np.load(sequence_path, allow_pickle=False).astype(np.float32)
                    valid_cache = cached_sequence.shape == (args.sequence_length, 128) and np.isfinite(cached_sequence).all()
                except (OSError, TypeError, ValueError):
                    valid_cache = False
            if not valid_cache:
                sequence = preprocessor.process_video(sample.video_path)
                np.save(sequence_path, sequence, allow_pickle=False)
            entries.append(
                json.dumps(
                    {
                        "sequence_path": str(sequence_path),
                        "label": sample.label,
                        "signer_id": sample.signer_id,
                        "sequence_length": args.sequence_length,
                        "feature_size": 128,
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        extractor.close()
    manifest_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    logger.info("Extracted {} landmark sequences", len(entries))


def _extract_data(args: argparse.Namespace) -> None:
    signers = [item.zfill(2) for item in _csv_values(args.signers)]
    paths = MindsDatasetDownloader(Path(args.input)).extract_signers(signers, Path(args.output))
    for path in paths:
        print(path)


def _read_landmark_samples(manifest_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_training_samples(entries: list[dict[str, object]], selected_classes: set[str], assigned_signers: set[str]) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    seen_paths: set[Path] = set()
    for entry in entries:
        try:
            sequence_path = Path(str(entry["sequence_path"])).expanduser().resolve()
            label = canonicalize_label(str(entry["label"]))
            signer_id = str(entry["signer_id"]).zfill(2)
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError("Manifest contains an invalid sample") from error
        if signer_id not in assigned_signers:
            raise ValueError(f"Manifest sample signer is not assigned to a split: {signer_id}")
        if sequence_path in seen_paths:
            raise ValueError(f"Manifest contains duplicate sequence path: {sequence_path}")
        seen_paths.add(sequence_path)
        if label in selected_classes:
            samples.append(DatasetSample(video_path=sequence_path, label=label, signer_id=signer_id))
    return samples


def _train(args: argparse.Namespace) -> None:
    settings = PipelineSettings(
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        augmentation_probability=args.augmentation_probability,
        augmentation_noise_std=args.augmentation_noise_std,
        device=args.device,
        random_seed=args.seed,
    )
    entries = _read_landmark_samples(Path(args.manifest))
    invalid_entries = [
        str(entry.get("sequence_path", "")) for entry in entries if entry.get("sequence_length") != settings.sequence_length or entry.get("feature_size") != 128
    ]
    if invalid_entries:
        raise ValueError(f"Manifest schema does not match requested sequence: {invalid_entries[0]}")
    selected_classes = list(dict.fromkeys(canonicalize_label(value) for value in _csv_values(args.classes)))
    if not selected_classes:
        raise ValueError("--classes is required for training")
    manifest_classes = {str(entry.get("label", "")) for entry in entries}
    missing_classes = sorted(set(selected_classes) - manifest_classes)
    if missing_classes:
        raise ValueError(f"Requested classes are absent from manifest: {', '.join(missing_classes)}")
    assigned_signers = _signer_set(args.train_signers) | _signer_set(args.validation_signers) | _signer_set(args.test_signers)
    samples = _build_training_samples(entries, set(selected_classes), assigned_signers)
    splits = build_signer_split(samples, _signer_set(args.train_signers), _signer_set(args.validation_signers), _signer_set(args.test_signers))
    labels = sorted(selected_classes)
    class_to_index = {label: index for index, label in enumerate(labels)}
    augmenter = LandmarkAugmenter(
        probability=settings.augmentation_probability,
        noise_std=settings.augmentation_noise_std,
        seed=settings.random_seed,
    )
    datasets = {
        name: LandmarkSequenceDataset(
            [sample.video_path for sample in split_samples],
            [class_to_index[sample.label] for sample in split_samples],
            expected_sequence_length=settings.sequence_length,
            transform=augmenter if name == "train" else None,
        )
        for name, split_samples in splits.items()
    }
    device = resolve_device(settings.device)
    set_seed(settings.random_seed)
    model = GRUSignClassifier(128, settings.hidden_size, len(labels), dropout=settings.dropout)
    model, history = train_model(
        model,
        datasets["train"],
        datasets["validation"],
        device,
        settings.epochs,
        settings.batch_size,
        settings.learning_rate,
        settings.random_seed,
        settings.weight_decay,
        settings.label_smoothing,
    )
    test_metrics = evaluate_model(model, datasets["test"], device, len(labels), settings.batch_size)
    save_checkpoint(
        Path(args.checkpoint),
        model,
        class_to_index,
        {
            "sequence_length": settings.sequence_length,
            "feature_size": 128,
            "device": str(device),
            "seed": settings.random_seed,
            "history": history,
            "test_metrics": test_metrics,
            "training": {
                "augmentation_probability": settings.augmentation_probability,
                "augmentation_noise_std": settings.augmentation_noise_std,
                "weight_decay": settings.weight_decay,
                "label_smoothing": settings.label_smoothing,
            },
            "split_signers": {
                "train": sorted(_signer_set(args.train_signers)),
                "validation": sorted(_signer_set(args.validation_signers)),
                "test": sorted(_signer_set(args.test_signers)),
            },
        },
    )
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


def _camera(args: argparse.Namespace) -> None:
    settings = PipelineSettings(
        sequence_length=args.sequence_length,
        confidence_threshold=args.confidence_threshold,
        stability_window_size=args.stability_window_size,
        minimum_consensus=args.minimum_consensus,
        cooldown_seconds=args.cooldown_seconds,
        camera_index=args.camera_index,
        device=args.device,
    )
    device = resolve_device(settings.device)
    app = RealtimeCaptionApp(
        checkpoint_path=Path(args.checkpoint),
        model_asset_path=Path(args.model_asset),
        camera_index=settings.camera_index,
        sequence_length=settings.sequence_length,
        confidence_threshold=settings.confidence_threshold,
        stability_window_size=settings.stability_window_size,
        minimum_consensus=settings.minimum_consensus,
        cooldown_seconds=settings.cooldown_seconds,
        device=device,
    )
    app.run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sinala")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data")
    inspect_parser.add_argument("--data-root", required=True)
    inspect_parser.add_argument("--classes", default="")
    inspect_parser.set_defaults(handler=_inspect_data)

    download_parser = subparsers.add_parser("download-data")
    download_parser.add_argument("--signers", required=True)
    download_parser.add_argument("--output", required=True)
    download_parser.set_defaults(handler=_download_data)

    extract_data_parser = subparsers.add_parser("extract-data")
    extract_data_parser.add_argument("--input", required=True)
    extract_data_parser.add_argument("--output", required=True)

    extract_data_parser.add_argument("--signers", required=True)
    extract_data_parser.set_defaults(handler=_extract_data)
    model_parser = subparsers.add_parser("download-model")
    model_parser.add_argument("--output", default="models/hand_landmarker.task")
    model_parser.set_defaults(handler=_download_model)

    extract_parser = subparsers.add_parser("extract-landmarks")
    extract_parser.add_argument("--data-root", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--classes", required=True)
    extract_parser.add_argument("--sequence-length", type=int, default=32)
    extract_parser.set_defaults(handler=_extract_landmarks)
    extract_parser.add_argument("--model-asset", default="models/hand_landmarker.task")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--manifest", required=True)
    train_parser.add_argument("--checkpoint", required=True)
    train_parser.add_argument("--classes", required=True)
    train_parser.add_argument("--train-signers", required=True)
    train_parser.add_argument("--validation-signers", required=True)
    train_parser.add_argument("--test-signers", required=True)
    train_parser.add_argument("--sequence-length", type=int, default=32)
    train_parser.add_argument("--hidden-size", type=int, default=128)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--weight-decay", type=float, default=0.0001)
    train_parser.add_argument("--label-smoothing", type=float, default=0.05)
    train_parser.add_argument("--augmentation-probability", type=float, default=0.8)
    train_parser.add_argument("--augmentation-noise-std", type=float, default=0.01)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    train_parser.set_defaults(handler=_train)

    camera_parser = subparsers.add_parser("camera")
    camera_parser.add_argument("--checkpoint", required=True)
    camera_parser.add_argument("--sequence-length", type=int, default=32)
    camera_parser.add_argument("--confidence-threshold", type=float, default=0.75)
    camera_parser.add_argument("--stability-window-size", type=int, default=5)
    camera_parser.add_argument("--minimum-consensus", type=int, default=4)
    camera_parser.add_argument("--cooldown-seconds", type=float, default=0.8)
    camera_parser.add_argument("--camera-index", type=int, default=0)
    camera_parser.add_argument("--model-asset", default="models/hand_landmarker.task")
    camera_parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    camera_parser.set_defaults(handler=_camera)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.handler(args)


__all__ = ["main"]
