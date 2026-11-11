
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "processed" / "sequences" / "split"
MODEL_PATH = ROOT / "models" / "upfall_lstm_best.keras"

X_test = np.load(DATA_DIR / "X_test_normalized.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

model = tf.keras.models.load_model(MODEL_PATH)

print("Test data shape:", X_test.shape)
print("Test label shape:", y_test.shape)

# Predicted probability for label 1
probabilities = model.predict(X_test, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Label 0", "Label 1"],
    zero_division=0,
))

if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
