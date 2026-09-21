import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

print("=" * 70)
print("UP-FALL LSTM FALL DETECTION TRAINING")
print("=" * 70)

# ============================================================
# 1. PATHS
# ============================================================

_HERE = Path(__file__).resolve().parent
_LEGACY = Path(r"D:\PatientSafetyAI")
_SPLIT = Path("processed") / "sequences" / "split" / "X_train_normalized.npy"
if (_HERE / _SPLIT).exists():
    BASE_DIR = _HERE
elif (_LEGACY / _SPLIT).exists():
    BASE_DIR = _LEGACY
else:
    BASE_DIR = _HERE
DATA_DIR = BASE_DIR / "processed" / "sequences" / "split"

X_TRAIN_PATH = DATA_DIR / "X_train_normalized.npy"
X_TEST_PATH = DATA_DIR / "X_test_normalized.npy"
Y_TRAIN_PATH = DATA_DIR / "y_train.npy"
Y_TEST_PATH = DATA_DIR / "y_test.npy"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading data...")

X_train = np.load(X_TRAIN_PATH)
X_test = np.load(X_TEST_PATH)
y_train = np.load(Y_TRAIN_PATH)
y_test = np.load(Y_TEST_PATH)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)

# ============================================================
# 3. CHECK DATA
# ============================================================

print("\n" + "-" * 70)
print("DATA CHECK")
print("-" * 70)

print("Training samples:", len(X_train))
print("Testing samples: ", len(X_test))
print("Time steps:      ", X_train.shape[1])
print("Features/frame:  ", X_train.shape[2])

print("\nTraining labels:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Label {label}: {count}")

print("\nTesting labels:")
unique, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Label {label}: {count}")

# ============================================================
# 4. HANDLE CLASS IMBALANCE
# ============================================================

print("\n" + "-" * 70)
print("CLASS WEIGHTS")
print("-" * 70)

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

for cls, weight in class_weights.items():
    print(f"Class {cls}: weight = {weight:.4f}")

# ============================================================
# 5. BUILD LSTM MODEL
# ============================================================

print("\n" + "-" * 70)
print("BUILDING LSTM MODEL")
print("-" * 70)

model = models.Sequential([
    layers.Input(shape=(20, 99)),

    layers.LSTM(
        96,
        return_sequences=True
    ),

    layers.Dropout(0.25),

    layers.LSTM(
        64
    ),

    layers.Dropout(0.25),

    layers.Dense(
        32,
        activation="relu"
    ),

    layers.Dropout(0.2),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=tf.keras.losses.BinaryFocalCrossentropy(
        gamma=2.0,
        alpha=0.75,
        from_logits=False,
    ),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# ============================================================
# 6. CALLBACKS
# ============================================================

best_model_path = MODEL_DIR / "upfall_lstm_best.keras"

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    filepath=best_model_path,
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

# ============================================================
# 7. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,

    validation_split=0.2,

    epochs=50,

    batch_size=32,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_lr
    ],

    verbose=1
)

# ============================================================
# 8. SAVE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "upfall_lstm_final.keras"

model.save(final_model_path)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nBest model:")
print(best_model_path)

print("\nFinal model:")
print(final_model_path)

print("\nThe model is now trained.")