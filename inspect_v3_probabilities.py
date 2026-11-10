import numpy as np
import tensorflow as tf

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "Datasets" / "processed" / "sequences" / "split_subjectwise_v2"
MODEL_PATH = ROOT / "models" / "upfall_lstm_v3_best.keras"

X_val = np.load(DATA_DIR / "X_val_normalized.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

model = tf.keras.models.load_model(MODEL_PATH)

prob = model.predict(X_val, verbose=0).ravel()

class_0 = prob[y_val == 0]
class_1 = prob[y_val == 1]

print("\nCLASS 0 PROBABILITIES")
print("Count:", len(class_0))
print("Mean:", np.mean(class_0))
print("Median:", np.median(class_0))
print("Min:", np.min(class_0))
print("Max:", np.max(class_0))

print("\nCLASS 1 PROBABILITIES")
print("Count:", len(class_1))
print("Mean:", np.mean(class_1))
print("Median:", np.median(class_1))
print("Min:", np.min(class_1))
print("Max:", np.max(class_1))

print("\nINDIVIDUAL VALIDATION PREDICTIONS")

for i in range(len(y_val)):
    print(
        f"Sample {i+1:02d} | "
        f"Actual: {y_val[i]} | "
        f"Probability: {prob[i]:.4f}"
    )