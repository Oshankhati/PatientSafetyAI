"""Train comparable subject-wise baseline and COCO-17 XY fall models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.utils.class_weight import compute_class_weight

from src.features.fall_dataset import load_fall_dataset_split
from src.temporal.threshold_tune import select_operating_point, sweep_thresholds


def _choose_validation_threshold(y_val: np.ndarray, p_class1: np.ndarray) -> dict:
    rows = sweep_thresholds(y_val, p_class1, np.round(np.arange(0.05, 0.951, 0.01), 2))
    return {"selected": select_operating_point(rows), "sweep": rows}


def _train_one(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_path: Path,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}

    model = models.Sequential(
        [
            layers.Input(shape=x_train.shape[1:]),
            layers.LSTM(64),
            layers.Dropout(0.30),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(1, activation="sigmoid"),
        ],
        name=f"fall_{name}",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-5,
                verbose=1,
            ),
        ],
        verbose=2,
    )
    probabilities = model.predict(x_val, verbose=0).ravel()
    operating_point = _choose_validation_threshold(y_val, probabilities)
    return {
        "name": name,
        "model_path": str(model_path.relative_to(ROOT)),
        "feature_shape": list(x_train.shape[1:]),
        "class_weights": class_weights,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "validation_operating_point": operating_point["selected"],
        "validation_threshold_sweep": operating_point["sweep"],
        "threshold_source": "SUBJECT4 validation only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_dir = ROOT / "results" / f"fall_experiment_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    experiments = {}
    for offset, representation in enumerate(("upfall33_xyz", "coco17_xy")):
        data = load_fall_dataset_split(representation, include_test=False)
        model_name = "baseline_99" if representation == "upfall33_xyz" else "coco17_xy"
        metrics = _train_one(
            model_name,
            data.x_train,
            data.y_train,
            data.x_val,
            data.y_val,
            experiment_dir / f"{model_name}.keras",
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed + offset,
        )
        metrics.update(
            {
                "representation": representation,
                "train_subjects": ["SUBJECT1", "SUBJECT2", "SUBJECT3"],
                "validation_subject": "SUBJECT4",
                "test_subject": "SUBJECT5 (not read or used during training)",
                "train_sequences": int(len(data.y_train)),
                "validation_sequences": int(len(data.y_val)),
                "train_class_counts": {
                    str(label): int(count)
                    for label, count in zip(*np.unique(data.y_train, return_counts=True))
                },
                "validation_class_counts": {
                    str(label): int(count)
                    for label, count in zip(*np.unique(data.y_val, return_counts=True))
                },
            }
        )
        experiments[model_name] = metrics

        if representation == "coco17_xy":
            stats_dir = ROOT / "processed" / "sequences" / "coco17_xy_split"
            stats_dir.mkdir(parents=True, exist_ok=True)
            np.save(stats_dir / "normalization_mean.npy", data.mean)
            np.save(stats_dir / "normalization_std.npy", data.std)
            shutil.copy2(
                experiment_dir / f"{model_name}.keras",
                ROOT / "models" / "coco17_xy_lstm_best.keras",
            )

    payload = {
        "protocol": (
            "Train on SUBJECT1-3; select checkpoint and threshold on SUBJECT4; "
            "SUBJECT5 excluded from training and selection."
        ),
        "test_set_caveat": (
            "Earlier project work evaluated the prior baseline and swept "
            "thresholds on SUBJECT5; it is not a pristine project-wide holdout."
        ),
        "test_evaluated": False,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "experiments": experiments,
    }
    (experiment_dir / "training.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    print(f"Experiment: {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
