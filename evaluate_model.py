import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("UP-FALL LSTM MODEL EVALUATION")
print("=" * 70)

BASE_DIR = Path(r"D:\PatientSafetyAI")

DATA_DIR = BASE_DIR / "processed" / "sequences" / "split"
MODEL_PATH = BASE_DIR / "models" / "upfall_lstm_best.keras"

# ------------------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------------------

print("\nLoading test data...")

X_test = np.load(DATA_DIR / "X_test_normalized.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ------------------------------------------------------------
# LOAD BEST MODEL
# ------------------------------------------------------------

print("\nLoading best LSTM model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("GENERATING PREDICTIONS")
print("-" * 70)

probabilities = model.predict(
    X_test,
    verbose=1
).ravel()

predictions = (probabilities >= 0.5).astype(int)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(y_test, predictions)

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-score : {f1:.4f} ({f1 * 100:.2f}%)")

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)

print("\n                Predicted")
print("              0        1")
print(f"Actual 0    {cm[0,0]:5d}    {cm[0,1]:5d}")
print(f"Actual 1    {cm[1,0]:5d}    {cm[1,1]:5d}")

# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Class 0", "Class 1"],
        zero_division=0
    )
)

# ------------------------------------------------------------
# PREDICTION DISTRIBUTION
# ------------------------------------------------------------

print("-" * 70)
print("PREDICTION DISTRIBUTION")
print("-" * 70)

unique, counts = np.unique(predictions, return_counts=True)

for label, count in zip(unique, counts):
    print(f"Predicted {label}: {count}")

print("\nActual distribution:")

unique, counts = np.unique(y_test, return_counts=True)

for label, count in zip(unique, counts):
    print(f"Actual {label}: {count}")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)