"""Load the UP-Fall LSTM baseline and run sequence inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.paths import resolve_path

logger = logging.getLogger(__name__)


class LstmPredictor:
    """Binary LSTM wrapper. Output is P(class=1) from a sigmoid head.

    In this project's cleaned UP-Fall labels:
      class 0 ≈ fall-phase (minority)
      class 1 ≈ non-fall

    Therefore ``fall_probability = 1 - P(class=1)`` when ``fall_class == 0``.
    """

    def __init__(
        self,
        model_path: str | Path = "models/upfall_lstm_best.keras",
        decision_threshold: float = 0.5,
        fall_class: int = 0,
    ) -> None:
        self.model_path = resolve_path(model_path)
        self.decision_threshold = decision_threshold
        self.fall_class = fall_class
        self._model: Any | None = None
        self.available = False
        self.load_error: str | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            self.load_error = (
                f"LSTM model not found at {self.model_path}. "
                "Train with: python train_lstm.py (uses processed sequences)."
            )
            logger.warning(self.load_error)
            self.available = False
            return
        try:
            import tensorflow as tf

            self._model = tf.keras.models.load_model(self.model_path)
            self.available = True
            logger.info("Loaded LSTM baseline from %s", self.model_path)
        except Exception as exc:
            self.load_error = f"Failed to load LSTM: {exc}"
            logger.error(self.load_error)
            self.available = False

    def predict_sequence(self, sequence: np.ndarray) -> dict[str, float | None]:
        """Predict on shape (T, 99). Returns fall_probability in [0, 1] or Nones."""
        if not self.available or self._model is None:
            return {
                "p_class1": None,
                "fall_probability": None,
                "predicted_class": None,
            }

        x = np.asarray(sequence, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 99:
            raise ValueError(f"Expected sequence shape (T, 99), got {x.shape}")
        batch = np.expand_dims(x, axis=0)
        try:
            p1 = float(np.asarray(self._model.predict(batch, verbose=0)).ravel()[0])
        except Exception as exc:
            logger.error("LSTM inference failed: %s", exc)
            return {
                "p_class1": None,
                "fall_probability": None,
                "predicted_class": None,
            }

        p1 = float(np.clip(p1, 0.0, 1.0))
        if self.fall_class == 0:
            fall_p = 1.0 - p1
            pred = 0 if p1 < self.decision_threshold else 1
        else:
            fall_p = p1
            pred = 1 if p1 >= self.decision_threshold else 0

        return {
            "p_class1": p1,
            "fall_probability": fall_p,
            "predicted_class": float(pred),
        }
