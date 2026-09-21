"""Evaluate the existing UP-Fall LSTM with relative paths; write versioned results."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import project_root


def _rate(cm: np.ndarray, positive: int = 0) -> dict[str, float]:
    """FPR/FNR treating ``positive`` as the class of interest (default: fall=0)."""
    # sklearn cm[i,j] = true i, pred j
    tn = fp = fn = tp = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == positive and j == positive:
                tp += cm[i, j]
            elif i == positive and j != positive:
                fn += cm[i, j]
            elif i != positive and j == positive:
                fp += cm[i, j]
            else:
                tn += cm[i, j]
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return {"false_positive_rate": float(fpr), "false_negative_rate": float(fnr)}


def main() -> int:
    root = project_root()
    data = root / "processed" / "sequences" / "split"
    model_path = root / "models" / "upfall_lstm_best.keras"
    if not model_path.exists():
        print(f"Model missing: {model_path}\nTrain with python train_lstm.py")
        return 1

    import tensorflow as tf

    x_test = np.load(data / "X_test_normalized.npy")
    y_test = np.load(data / "y_test.npy")
    model = tf.keras.models.load_model(model_path)
    p1 = model.predict(x_test, verbose=0).ravel()
    pred = (p1 >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(
        y_test, pred, target_names=["Class 0 (fall)", "Class 1 (non-fall)"], zero_division=0
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / "results" / f"lstm_baseline_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    unique, counts = np.unique(y_test, return_counts=True)
    metrics = {
        "model": str(model_path.relative_to(root)),
        "input_shape": list(x_test.shape),
        "protocol": "subject-wise holdout SUBJECT5",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "test_class_counts": {int(k): int(v) for k, v in zip(unique, counts)},
        "fall_class_0_rates": _rate(cm, positive=0),
        "note": "Baseline LSTM. Domain of Kinect UP-Fall sequences, not live YOLO.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
