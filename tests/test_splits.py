from pathlib import Path

import pytest

from sinala.data.dataset_sample import DatasetSample
from sinala.data.splits import build_signer_split


def _sample(label: str, signer_id: str) -> DatasetSample:
    return DatasetSample(video_path=Path(f"{signer_id}_{label}.npy"), label=label, signer_id=signer_id)


def test_build_signer_split_when_signers_are_disjoint_then_creates_partitions() -> None:
    samples = [_sample(label, signer) for signer in ("01", "02", "03") for label in ("banco", "vacina")]

    result = build_signer_split(samples, {"01"}, {"02"}, {"03"})

    assert {sample.signer_id for sample in result["train"]} == {"01"}
    assert {sample.signer_id for sample in result["validation"]} == {"02"}
    assert {sample.signer_id for sample in result["test"]} == {"03"}


def test_build_signer_split_when_signers_overlap_then_rejects_split() -> None:
    samples = [_sample("banco", signer) for signer in ("01", "02", "03")]

    with pytest.raises(ValueError, match="overlap"):
        build_signer_split(samples, {"01"}, {"01"}, {"03"})


def test_build_signer_split_when_class_is_missing_from_partition_then_rejects_split() -> None:
    samples = [_sample("banco", "01"), _sample("vacina", "02"), _sample("banco", "03"), _sample("vacina", "03")]

    with pytest.raises(ValueError, match="missing classes"):
        build_signer_split(samples, {"01"}, {"02"}, {"03"})
