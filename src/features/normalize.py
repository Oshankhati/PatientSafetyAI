"""Apply UP-Fall **training-set** mean/std. Never fit on live or test data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.utils.paths import project_root, resolve_path

logger = logging.getLogger(__name__)


class TrainStatNormalizer:
    """Z-score using saved training statistics (shape broadcastable to (T, 99))."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)
        if self.mean.shape != self.std.shape:
            raise ValueError("mean/std feature width mismatch")
        self.std = np.where(self.std == 0, 1.0, self.std)

    @classmethod
    def from_files(
        cls,
        mean_path: str | Path | None = None,
        std_path: str | Path | None = None,
    ) -> TrainStatNormalizer:
        root = project_root()
        split = root / "processed" / "sequences" / "split"
        mean_file = resolve_path(mean_path) if mean_path else split / "normalization_mean.npy"
        std_file = resolve_path(std_path) if std_path else split / "normalization_std.npy"
        if not mean_file.exists() or not std_file.exists():
            raise FileNotFoundError(
                "Training normalization files not found. Expected "
                f"{mean_file} and {std_file}. Re-run normalize_data.py on the "
                "subject-wise training split only."
            )
        mean = np.load(mean_file)
        std = np.load(std_file)
        logger.info("Loaded training normalization stats from %s", split)
        return cls(mean, std)

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize a sequence of shape (T, F) or (N, T, F)."""
        x = np.asarray(sequence, dtype=np.float32)
        if x.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"Feature width {x.shape[-1]} != stats width {self.mean.shape[0]}"
            )
        return (x - self.mean) / self.std
