import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 70)
print("UP-FALL SUBJECT-WISE DATASET SPLIT")
print("=" * 70)

BASE_DIR = Path(r"D:\PatientSafetyAI\processed\sequences")

X_PATH = BASE_DIR / "X_sequences.npy"
Y_PATH = BASE_DIR / "y_sequences.npy"
META_PATH = BASE_DIR / "sequence_metadata.csv"

OUTPUT_DIR = BASE_DIR / "split"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

X = np.load(X_PATH)
y = np.load(Y_PATH)
metadata = pd.read_csv(META_PATH)

print("\nOriginal dataset:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Metadata rows: {len(metadata)}")

# ------------------------------------------------------------
# CHECK DATA ALIGNMENT
# ------------------------------------------------------------

if len(X) != len(y) or len(X) != len(metadata):
    raise ValueError("X, y and metadata do not have the same number of samples.")

# ------------------------------------------------------------
# SUBJECT-WISE SPLIT
# ------------------------------------------------------------

TEST_SUBJECT = "SUBJECT5"

test_mask = metadata["subject"] == TEST_SUBJECT
train_mask = metadata["subject"] != TEST_SUBJECT

X_train = X[train_mask.values]
y_train = y[train_mask.values]

X_test = X[test_mask.values]
y_test = y[test_mask.values]

train_metadata = metadata[train_mask].copy()
test_metadata = metadata[test_mask].copy()

# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("SPLIT RESULTS")
print("-" * 70)

print("\nTraining subjects:")
print(train_metadata["subject"].value_counts().sort_index())

print("\nTesting subjects:")
print(test_metadata["subject"].value_counts().sort_index())

print("\nTraining data:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print("\nTesting data:")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")

print("\n" + "-" * 70)
print("LABEL DISTRIBUTION")
print("-" * 70)

print("\nTraining labels:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTesting labels:")
print(pd.Series(y_test).value_counts().sort_index())

# ------------------------------------------------------------
# SAVE SPLIT DATA
# ------------------------------------------------------------

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "y_train.npy", y_train)

np.save(OUTPUT_DIR / "X_test.npy", X_test)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

train_metadata.to_csv(
    OUTPUT_DIR / "train_metadata.csv",
    index=False
)

test_metadata.to_csv(
    OUTPUT_DIR / "test_metadata.csv",
    index=False
)

print("\n" + "-" * 70)
print("FILES SAVED")
print("-" * 70)

print(f"\n{OUTPUT_DIR}")

print("\n[OK] X_train.npy")
print("[OK] y_train.npy")
print("[OK] X_test.npy")
print("[OK] y_test.npy")
print("[OK] train_metadata.csv")
print("[OK] test_metadata.csv")

print("\n" + "=" * 70)
print("SUBJECT-WISE SPLIT COMPLETE")
print("=" * 70)