"""Sweep LSTM decision threshold on SUBJECT5 (subject-wise holdout).

Does not retrain and does not overwrite the baseline experiment folder.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.temporal.threshold_tune import select_operating_point, sweep_thresholds
from src.utils.paths import project_root


def main() -> int:
    root = project_root()
    data = root / "processed" / "sequences" / "split"
    model_path = root / "models" / "upfall_lstm_best.keras"
    x_path = data / "X_test_normalized.npy"
    y_path = data / "y_test.npy"
    if not model_path.exists():
        print(f"Model missing: {model_path}\nTrain with python train_lstm.py")
        return 1
    if not x_path.exists() or not y_path.exists():
        print(f"Test arrays missing under {data}")
        return 1

    import tensorflow as tf

    x_test = np.load(x_path)
    y_test = np.load(y_path)
    model = tf.keras.models.load_model(model_path)
    p1 = model.predict(x_test, verbose=0).ravel()
    rows = sweep_thresholds(y_test, p1)
    chosen = select_operating_point(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / "results" / f"lstm_threshold_sweep_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(model_path.relative_to(root)),
        "protocol": "subject-wise holdout SUBJECT5; threshold sweep only (no retraining)",
        "input_shape": list(x_test.shape),
        "baseline_threshold": 0.5,
        "selected": chosen,
        "sweep": rows,
        "note": (
            "Class 0 = fall-phase (minority). Raising the class-1 threshold "
            "increases fall recall. Not clinically validated."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "threshold  acc    f1_macro  fall_recall  fall_prec  FPR    FNR",
    ]
    for r in rows:
        lines.append(
            f"{r['threshold']:.2f}       {r['accuracy']:.3f}  {r['f1_macro']:.3f}     "
            f"{r['fall_recall']:.3f}        {r['fall_precision']:.3f}     "
            f"{r['false_positive_rate']:.3f}  {r['false_negative_rate']:.3f}"
        )
    lines.append(f"\nSelected operating point: {json.dumps(chosen, indent=2)}")
    report = "\n".join(lines) + "\n"
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
