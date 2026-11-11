
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("=" * 70)
print("PATIENT SAFETY AI - LSTM EXPERIMENT V2")
print("=" * 70)

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    ROOT / "Datasets" / "processed" /
    "sequences" / "split_subjectwise_v2"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "upfall_lstm_v2_best.keras"
FINAL_MODEL = MODEL_DIR / "upfall_lstm_v2_final.keras"

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

X_train = np.load(DATA_DIR / "X_train_normalized.npy")
X_val = np.load(DATA_DIR / "X_val_normalized.npy")
X_test = np.load(DATA_DIR / "X_test_normalized.npy")

y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("\nDATA SHAPES")
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)

# Test data is deliberately not used during training.

# ------------------------------------------------------------
# 3. CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\nTRAINING LABEL DISTRIBUTION")
print(np.unique(y_train, return_counts=True))

print("\nVALIDATION LABEL DISTRIBUTION")
print(np.unique(y_val, return_counts=True))

# ------------------------------------------------------------
# 4. CLASS WEIGHTS
# ------------------------------------------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nCLASS WEIGHTS")
print(class_weights)

# ------------------------------------------------------------
# 5. BUILD MODEL
# ------------------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(20, 99)),

    layers.LSTM(96, return_sequences=True),
    layers.Dropout(0.25),

    layers.LSTM(64),
    layers.Dropout(0.25),

    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=False,
        gamma=2.0,
        from_logits=False
    ),

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# ------------------------------------------------------------
# 6. CALLBACKS
# ------------------------------------------------------------

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    filepath=BEST_MODEL,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# ------------------------------------------------------------
# 7. TRAIN
# ------------------------------------------------------------

print("\nSTARTING TRAINING")

history = model.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=50,
    batch_size=32,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ],

    shuffle=True,
    verbose=1
)

# ------------------------------------------------------------
# 8. SAVE FINAL MODEL
# ------------------------------------------------------------

model.save(FINAL_MODEL)

print("\nTRAINING COMPLETED")
print("Best model:", BEST_MODEL)
print("Final model:", FINAL_MODEL)

# ------------------------------------------------------------
# 9. VALIDATION EVALUATION
# ------------------------------------------------------------

best_model = tf.keras.models.load_model(BEST_MODEL)

val_prob = best_model.predict(X_val, verbose=0).ravel()
val_pred = (val_prob >= 0.5).astype(int)

print("\nVALIDATION CONFUSION MATRIX")
print(confusion_matrix(y_val, val_pred))

print("\nVALIDATION CLASSIFICATION REPORT")
print(classification_report(
    y_val,
    val_pred,
    zero_division=0
))

print("\nVALIDATION ROC-AUC")
print(roc_auc_score(y_val, val_prob))

print("\nIMPORTANT:")
print("SUBJECT5 has not been used for model selection.")
print("Its final evaluation will be performed separately.")
