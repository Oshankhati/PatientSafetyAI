"""Leakage-aware subject-wise UP-Fall splits for fall-model experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.coco17_xy import upfall99_to_coco17_xy
from src.utils.paths import project_root

TRAIN_SUBJECTS = ("SUBJECT1", "SUBJECT2", "SUBJECT3")
VALIDATION_SUBJECT = "SUBJECT4"
TEST_SUBJECT = "SUBJECT5"


@dataclass(frozen=True)
class FallDatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    train_metadata: pd.DataFrame
    val_metadata: pd.DataFrame
    test_metadata: pd.DataFrame


def load_fall_dataset_split(
    representation: str,
    root: Path | None = None,
    *,
    include_test: bool = True,
) -> FallDatasetSplit:
    """Split by subject, fit on training only, and optionally leave test untouched."""
    project = root or project_root()
    data_dir = project / "processed" / "sequences"
    x = np.load(data_dir / "X_sequences.npy", mmap_mode="r")
    y = np.load(data_dir / "y_sequences.npy", mmap_mode="r").reshape(-1)
    metadata = pd.read_csv(
        data_dir / "sequence_metadata.csv",
        usecols=(
            None
            if include_test
            else ["subject", "file", "start_frame", "end_frame"]
        ),
    )

    if x.ndim != 3 or x.shape[1:] != (20, 99):
        raise ValueError(f"Expected UP-Fall sequences shaped (N, 20, 99), got {x.shape}")
    if len(x) != len(y) or len(x) != len(metadata):
        raise ValueError("X, y and sequence metadata lengths do not match")
    if "subject" not in metadata or (include_test and "label" not in metadata):
        raise ValueError("Sequence metadata is missing required columns")
    subjects = set(metadata["subject"].astype(str))
    expected = set(TRAIN_SUBJECTS) | {VALIDATION_SUBJECT, TEST_SUBJECT}
    if subjects != expected:
        raise ValueError(f"Expected subjects {sorted(expected)}, found {sorted(subjects)}")

    if representation not in {"upfall33_xyz", "coco17_xy"}:
        raise ValueError(f"Unsupported representation: {representation}")

    subject_values = metadata["subject"].astype(str).to_numpy()
    train_indices = np.flatnonzero(np.isin(subject_values, TRAIN_SUBJECTS))
    val_indices = np.flatnonzero(subject_values == VALIDATION_SUBJECT)
    test_indices = np.flatnonzero(subject_values == TEST_SUBJECT)
    train_mask = np.isin(subject_values, TRAIN_SUBJECTS)
    val_mask = subject_values == VALIDATION_SUBJECT
    test_mask = subject_values == TEST_SUBJECT
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Train, validation, and test splits must all be non-empty")

    if include_test:
        checked_indices = np.concatenate((train_indices, val_indices, test_indices))
        labels = metadata["label"].to_numpy(dtype=np.int32)[checked_indices]
        if not np.array_equal(labels, np.asarray(y[checked_indices], dtype=np.int32)):
            raise ValueError("Sequence metadata labels do not match y_sequences.npy")

    def select_features(indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(x[indices], dtype=np.float32)
        if representation == "coco17_xy":
            return upfall99_to_coco17_xy(selected)
        return selected

    x_train = select_features(train_indices)
    x_val = select_features(val_indices)
    x_test = (
        select_features(test_indices)
        if include_test
        else np.empty((0, x.shape[1], 34 if representation == "coco17_xy" else 99), np.float32)
    )
    y_train = np.asarray(y[train_indices], dtype=np.int32)
    y_val = np.asarray(y[val_indices], dtype=np.int32)
    y_test = (
        np.asarray(y[test_indices], dtype=np.int32)
        if include_test
        else np.empty((0,), dtype=np.int32)
    )
    mean = x_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = x_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std[std < 1e-8] = 1.0

    return FallDatasetSplit(
        x_train=((x_train - mean) / std).astype(np.float32),
        y_train=y_train,
        x_val=((x_val - mean) / std).astype(np.float32),
        y_val=y_val,
        x_test=((x_test - mean) / std).astype(np.float32),
        y_test=y_test,
        mean=mean,
        std=std,
        train_metadata=metadata.loc[train_mask].reset_index(drop=True),
        val_metadata=metadata.loc[val_mask].reset_index(drop=True),
        test_metadata=(
            metadata.loc[test_mask].reset_index(drop=True)
            if include_test
            else metadata.iloc[0:0].copy()
        ),
    )
