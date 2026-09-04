from collections.abc import Iterable

from sinala.data.dataset_sample import DatasetSample

SPLIT_NAMES = ("train", "validation", "test")


def build_signer_split(
    samples: Iterable[DatasetSample],
    train_signers: set[str],
    validation_signers: set[str],
    test_signers: set[str],
) -> dict[str, list[DatasetSample]]:
    signer_sets = {
        "train": set(train_signers),
        "validation": set(validation_signers),
        "test": set(test_signers),
    }
    validate_signer_split(signer_sets)
    sample_list = list(samples)
    labels = {sample.label for sample in sample_list}
    result = {name: [sample for sample in sample_list if sample.signer_id in signers] for name, signers in signer_sets.items()}
    for split_name, split_samples in result.items():
        split_labels = {sample.label for sample in split_samples}
        missing_labels = sorted(labels - split_labels)
        if missing_labels:
            raise ValueError(f"Split {split_name} is missing classes: {', '.join(missing_labels)}")
    return result


def validate_signer_split(signer_sets: dict[str, set[str]]) -> None:
    if set(signer_sets) != set(SPLIT_NAMES):
        raise ValueError(f"Signer split must contain exactly: {', '.join(SPLIT_NAMES)}")
    if any(not signer_sets[name] for name in SPLIT_NAMES):
        raise ValueError("Every split must contain at least one signer")
    for first_index, first_name in enumerate(SPLIT_NAMES):
        for second_name in SPLIT_NAMES[first_index + 1 :]:
            overlap = signer_sets[first_name] & signer_sets[second_name]
            if overlap:
                raise ValueError(f"Signer overlap between {first_name} and {second_name}: {', '.join(sorted(overlap))}")


__all__ = ["build_signer_split", "validate_signer_split"]
