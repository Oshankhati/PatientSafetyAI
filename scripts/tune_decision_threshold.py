"""Tune a COCO-17 model threshold on validation data only; never load SUBJECT5."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.fall_dataset import load_fall_dataset_split
from src.temporal.threshold_tune import select_operating_point, sweep_thresholds


def main() -> int:
    model_path = ROOT / "models" / "coco17_xy_lstm_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model missing: {model_path}\n"
            "Train it with: python scripts/train_fall_experiment.py"
        )

    import tensorflow as tf

    data = load_fall_dataset_split("coco17_xy", include_test=False)
    model = tf.keras.models.load_model(model_path)
    p_class1 = model.predict(data.x_val, verbose=0).ravel()
    rows = sweep_thresholds(
        data.y_val,
        p_class1,
        thresholds=np.round(np.arange(0.05, 0.951, 0.01), 2),
    )
    selected = select_operating_point(rows)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / f"coco17_validation_threshold_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "model": str(model_path.relative_to(ROOT)),
        "protocol": "SUBJECT4 validation only; SUBJECT5 test samples and labels not read",
        "validation_sequences": int(len(data.y_val)),
        "validation_class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(data.y_val, return_counts=True))
        },
        "selected": selected,
        "sweep": rows,
        "note": (
            "Class 0 is fall. If constraints_met is false, no tested threshold "
            "met both validation constraints; do not claim that it did."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
