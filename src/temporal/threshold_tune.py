"""Subject-wise decision-threshold sweep for the binary LSTM.

The baseline uses 0.5 on P(class=1). Class 0 is the minority fall phase.
Raising the class-1 threshold labels more sequences as class 0 (fall),
which can raise fall recall at the cost of more false alarms.

This is **not** a new trained class. It only changes the operating point
on the existing sigmoid head. Threshold selection must use validation data only. Final held-out test
metrics are computed by the separate evaluation script after model selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def apply_threshold(p_class1: np.ndarray, threshold: float) -> np.ndarray:
    """Map P(class=1) to labels with ``pred = 1`` iff ``p >= threshold``."""
    p = np.asarray(p_class1, dtype=np.float64).ravel()
    return (p >= float(threshold)).astype(int)


def _fall_rates(cm: np.ndarray) -> dict[str, float]:
    """Treat class 0 as the positive (fall) class."""
    tn = fp = fn = tp = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == 0 and j == 0:
                tp += int(cm[i, j])
            elif i == 0 and j != 0:
                fn += int(cm[i, j])
            elif i != 0 and j == 0:
                fp += int(cm[i, j])
            else:
                tn += int(cm[i, j])
    return {
        "fall_recall": float(tp / max(tp + fn, 1)),
        "fall_precision": float(tp / max(tp + fp, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
    }


def metrics_at_threshold(
    y_true: np.ndarray,
    p_class1: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y = np.asarray(y_true).ravel().astype(int)
    pred = apply_threshold(p_class1, threshold)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    unique, counts = np.unique(y, return_counts=True)
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_class1": float(precision_score(y, pred, zero_division=0)),
        "recall_class1": float(recall_score(y, pred, zero_division=0)),
        "f1_class1": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "class_counts": {int(k): int(v) for k, v in zip(unique, counts)},
    }
    out.update(_fall_rates(cm))
    return out


def sweep_thresholds(
    y_true: np.ndarray,
    p_class1: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if thresholds is None:
        thresholds = np.round(np.linspace(0.20, 0.80, 13), 2)
    return [metrics_at_threshold(y_true, p_class1, float(t)) for t in thresholds]


def select_operating_point(
    rows: list[dict[str, Any]],
    *,
    max_false_positive_rate: float = 0.50,
    min_fall_recall: float = 0.40,
) -> dict[str, Any]:
    """Pick a feasible row by fall F1, or explicitly mark fallback as infeasible.

    Preference: keep FPR (class 0 false alarms on non-fall) below the cap
    while meeting a minimum fall recall. If none qualify, retain the default
    0.5 threshold and report that the constraints were not met.
    """
    feasible = [
        r
        for r in rows
        if r["false_positive_rate"] <= max_false_positive_rate
        and r["fall_recall"] >= min_fall_recall
    ]
    if not rows:
        raise ValueError("At least one threshold result is required")
    if feasible:
        selected = max(
            feasible,
            key=lambda r: (
                2 * r["fall_precision"] * r["fall_recall"]
                / max(r["fall_precision"] + r["fall_recall"], 1e-12),
                r["fall_recall"],
                r["fall_precision"],
            ),
        )
        return {
            **selected,
            "constraints_met": True,
            "selection_reason": "highest fall F1 among feasible thresholds",
        }
    selected = min(rows, key=lambda r: abs(r["threshold"] - 0.5))
    return {
        **selected,
        "constraints_met": False,
        "selection_reason": "no feasible threshold; retained the default 0.5 threshold",
    }
