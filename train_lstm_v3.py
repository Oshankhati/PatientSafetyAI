import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Datasets" / "processed" / "sequences" / "split_subjectwise_v2"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "upfall_lstm_v3_best.keras"
FINAL_MODEL = MODEL_DIR / "upfall_lstm_v3_final.keras"

# Never overwrite an earlier V3 experiment accidentally.
if BEST_MODEL.exists() or FINAL_MODEL.exists():
    raise FileExistsError(
        "A V3 model file already exists. Back it up or rename it before rerunning."
    )

print("=" * 70)
print("PATIENT SAFETY AI - LSTM EXPERIMENT V3")
print("=" * 70)

# Load the established subject-wise split.
X_train = np.load(DATA_DIR / "X_train_normalized.npy")
X_val = np.load(DATA_DIR / "X_val_normalized.npy")
y_train = np.load(DATA_DIR / "y_train.npy").astype(np.int32)
y_val = np.load(DATA_DIR / "y_val.npy").astype(np.int32)

print("\nDATA SHAPES")
print("Training:  ", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)

if X_train.ndim != 3 or X_train.shape[1:] != (20, 99):
    raise ValueError(f"Expected X_train shape (samples, 20, 99); got {X_train.shape}")
if X_val.ndim != 3 or X_val.shape[1:] != (20, 99):
    raise ValueError(f"Expected X_val shape (samples, 20, 99); got {X_val.shape}")
if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
    raise ValueError("Training and validation must both contain labels 0 and 1.")

print("\nLABEL DISTRIBUTIONS")
print("Training:  ", np.unique(y_train, return_counts=True))
print("Validation:", np.unique(y_val, return_counts=True))

# Balanced weights are calculated from training labels only.
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
print("\nCLASS WEIGHTS")
print(class_weights)

# Smaller architecture than V2.
model = models.Sequential(
    [
        layers.Input(shape=(20, 99)),
        layers.LSTM(64),
        layers.Dropout(0.30),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="upfall_lstm_v3",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    ],
)

print("\nMODEL SUMMARY")
model.summary()

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)
checkpoint = callbacks.ModelCheckpoint(
    filepath=BEST_MODEL,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)