"""Evaluate finalized fall-model experiments once on the held-out SUBJECT5 set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.features.fall_dataset import load_fall_dataset_split
from src.temporal.threshold_tune import metrics_at_threshold


def _evaluate_model(experiment_dir: Path, name: str, setup: dict) -> dict:
    import tensorflow as tf

    representation = setup["representation"]
    data = load_fall_dataset_split(representation, root=ROOT)
    model_path = experiment_dir / f"{name}.keras"
    model = tf.keras.models.load_model(model_path)
    p_class1 = model.predict(data.x_test, verbose=0).ravel()
    threshold = float(setup["validation_operating_point"]["threshold"])
    row = metrics_at_threshold(data.y_test, p_class1, threshold)
    prediction = (p_class1 >= threshold).astype(np.int32)
    cm = confusion_matrix(data.y_test, prediction, labels=[0, 1])
    report = classification_report(
        data.y_test,
        prediction,
        labels=[0, 1],
        target_names=["Fall (class 0)", "Non-fall (class 1)"],
        zero_division=0,
    )
    result = {
        "name": name,
        "representation": representation,
        "model": str(model_path.relative_to(ROOT)),
        "parameters": int(model.count_params()),
        "test_subject": "SUBJECT5",
        "test_sequences": int(len(data.y_test)),
        "test_class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(data.y_test, return_counts=True))
        },
        "threshold": threshold,
        "threshold_source": "SUBJECT4 validation only",
        "threshold_constraints_met_on_validation": bool(
            setup["validation_operating_point"]["constraints_met"]
        ),
        "accuracy": float(accuracy_score(data.y_test, prediction)),
        "fall_precision": float(precision_score(data.y_test, prediction, pos_label=0, zero_division=0)),
        "fall_recall": float(recall_score(data.y_test, prediction, pos_label=0, zero_division=0)),
        "fall_f1": float(f1_score(data.y_test, prediction, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(data.y_test, prediction, average="macro", zero_division=0)),
        "fall_pr_auc": float(
            average_precision_score(data.y_test == 0, 1.0 - p_class1)
        ),
        "false_positive_rate": row["false_positive_rate"],
        "false_negative_rate": row["false_negative_rate"],
        "confusion_matrix_labels": ["fall", "non_fall"],
        "confusion_matrix": cm.tolist(),
    }
    return {"metrics": result, "report": report, "model": model}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to a fall_experiment_* directory created by train_fall_experiment.py",
    )
    args = parser.parse_args()
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        experiment_dir = ROOT / experiment_dir
    experiment_dir = experiment_dir.resolve()
    training_file = experiment_dir / "training.json"
    if not training_file.exists():
        parser.error(f"Experiment metadata not found: {training_file}")
    training = json.loads(training_file.read_text(encoding="utf-8"))
    if training.get("test_evaluated"):
        parser.error("This experiment has already been marked as test-evaluated.")

    results = {}
    summary_rows = []
    for name, setup in training["experiments"].items():
        evaluated = _evaluate_model(experiment_dir, name, setup)
        results[name] = evaluated
        prefix = experiment_dir / f"{name}_test"
        prefix.with_suffix(".json").write_text(
            json.dumps(evaluated["metrics"], indent=2), encoding="utf-8"
        )
        prefix.with_name(f"{name}_classification_report.txt").write_text(
            evaluated["report"], encoding="utf-8"
        )
        prefix.with_name(f"{name}_confusion_matrix.csv").write_text(
            "actual,predicted_fall,predicted_non_fall\n"
            f"fall,{evaluated['metrics']['confusion_matrix'][0][0]},"
            f"{evaluated['metrics']['confusion_matrix'][0][1]}\n"
            f"non_fall,{evaluated['metrics']['confusion_matrix'][1][0]},"
            f"{evaluated['metrics']['confusion_matrix'][1][1]}\n",
            encoding="utf-8",
        )
        validation = setup["validation_operating_point"]
        summary_rows.append(
            {
                "model": name,
                "parameters": evaluated["metrics"]["parameters"],
                "validation_threshold": validation["threshold"],
                "validation_fall_precision": validation["fall_precision"],
                "validation_fall_recall": validation["fall_recall"],
                "validation_fall_f1": (
                    2 * validation["fall_precision"] * validation["fall_recall"]
                    / max(validation["fall_precision"] + validation["fall_recall"], 1e-12)
                ),
                "test_accuracy": evaluated["metrics"]["accuracy"],
                "test_fall_precision": evaluated["metrics"]["fall_precision"],
                "test_fall_recall": evaluated["metrics"]["fall_recall"],
                "test_fall_f1": evaluated["metrics"]["fall_f1"],
                "test_fall_pr_auc": evaluated["metrics"]["fall_pr_auc"],
                "test_false_positive_rate": evaluated["metrics"]["false_positive_rate"],
            }
        )

    comparison_path = experiment_dir / "model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": training["protocol"],
        "test_subject": "SUBJECT5",
        "test_set_caveat": (
            "SUBJECT5 was excluded from training, checkpoint selection, and "
            "threshold selection in this experiment, but earlier project work "
            "evaluated the prior baseline and swept thresholds on SUBJECT5. "
            "Treat these results as exploratory, not as a pristine project-wide "
            "final holdout."
        ),
        "models": {name: item["metrics"] for name, item in results.items()},
    }
    (experiment_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (experiment_dir / "model_comparison.csv").resolve()
    training["test_evaluated"] = True
    (experiment_dir / "training.json").write_text(
        json.dumps(training, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Experiment report: {experiment_dir / 'test_metrics.json'}")
    print(f"Model comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
