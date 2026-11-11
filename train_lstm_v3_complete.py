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

if BEST_MODEL.exists() or FINAL_MODEL.exists():
    raise FileExistsError("V3 model output already exists; back it up or rename it before rerunning.")

print("=" * 70, flush=True)
print("PATIENT SAFETY AI - LSTM EXPERIMENT V3", flush=True)
print("=" * 70, flush=True)

# SUBJECT1-3 train; SUBJECT4 validation. SUBJECT5 is not loaded.
X_train = np.load(DATA_DIR / "X_train_normalized.npy")
X_val = np.load(DATA_DIR / "X_val_normalized.npy")
y_train = np.load(DATA_DIR / "y_train.npy").astype(np.int32)
y_val = np.load(DATA_DIR / "y_val.npy").astype(np.int32)

print("\nDATA SHAPES", flush=True)
print("Training:", X_train.shape, y_train.shape, flush=True)
print("Validation:", X_val.shape, y_val.shape, flush=True)

if X_train.ndim != 3 or X_train.shape[1:] != (20, 99):
    raise ValueError(f"Unexpected training shape: {X_train.shape}")
if X_val.ndim != 3 or X_val.shape[1:] != (20, 99):
    raise ValueError(f"Unexpected validation shape: {X_val.shape}")
if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
    raise ValueError("Training and validation labels must both contain 0 and 1.")

print("\nLABEL DISTRIBUTIONS", flush=True)
print("Training:", np.unique(y_train, return_counts=True), flush=True)
print("Validation:", np.unique(y_val, return_counts=True), flush=True)

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
print("\nCLASS WEIGHTS:", class_weights, flush=True)

model = models.Sequential([
    layers.Input(shape=(20, 99)),
    layers.LSTM(64),
    layers.Dropout(0.30),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(1, activation="sigmoid"),
], name="upfall_lstm_v3")

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

print("\nMODEL SUMMARY", flush=True)
model.summary()

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)
checkpoint = callbacks.ModelCheckpoint(
    filepath=BEST_MODEL, monitor="val_loss", mode="min",
    save_best_only=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4,
    min_lr=0.00001, verbose=1
)

print("\nSTARTING V3 TRAINING", flush=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    shuffle=True,
    verbose=1,
)

model.save(FINAL_MODEL)
print("\nTRAINING COMPLETED", flush=True)
print("Best model:", BEST_MODEL, flush=True)
print("Final model:", FINAL_MODEL, flush=True)

best_model = tf.keras.models.load_model(BEST_MODEL)
val_prob = best_model.predict(X_val, verbose=0).ravel()
val_pred = (val_prob >= 0.5).astype(np.int32)

print("\nV3 VALIDATION CONFUSION MATRIX (threshold=0.5)", flush=True)
print(confusion_matrix(y_val, val_pred), flush=True)
print("\nV3 VALIDATION CLASSIFICATION REPORT", flush=True)
print(classification_report(y_val, val_pred, labels=[0, 1], zero_division=0), flush=True)
print("\nV3 VALIDATION ROC-AUC", flush=True)
print(roc_auc_score(y_val, val_prob), flush=True)
print("\nSUBJECT5 was not loaded or used for model selection/evaluation.", flush=True)
