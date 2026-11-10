
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 65)
print("PATIENT SAFETY AI - EXPERIMENTAL DATA PREPARATION")
print("=" * 65)

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Datasets" / "processed" / "sequences"
OUTPUT_DIR = DATA_DIR / "split_subjectwise_v2"

# Load original data
print("\nLoading original sequences...")

X = np.load(DATA_DIR / "X_sequences.npy")
y = np.load(DATA_DIR / "y_sequences.npy")
metadata = pd.read_csv(DATA_DIR / "sequence_metadata.csv")

if not (len(X) == len(y) == len(metadata)):
    raise ValueError("X, y and metadata lengths do not match.")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Metadata rows:", len(metadata))

# Define subject-wise split
train_subjects = ["SUBJECT1", "SUBJECT2", "SUBJECT3"]
val_subject = "SUBJECT4"
test_subject = "SUBJECT5"

train_mask = metadata["subject"].isin(train_subjects).values
val_mask = (metadata["subject"] == val_subject).values
test_mask = (metadata["subject"] == test_subject).values

if not np.all(train_mask | val_mask | test_mask):
    raise ValueError("Some sequences have unexpected subject values.")

# Separate datasets
X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

train_meta = metadata[train_mask].copy()
val_meta = metadata[val_mask].copy()
test_meta = metadata[test_mask].copy()

# Display split
print("\nSUBJECT-WISE SPLIT")
print("Training:", train_meta["subject"].unique())
print("Validation:", val_meta["subject"].unique())
print("Testing:", test_meta["subject"].unique())

print("\nDATA SHAPES")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nLABEL DISTRIBUTIONS")
for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test)
]:
    print(f"\n{name}:")
    print(pd.Series(labels).value_counts().sort_index())

if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError("One of the splits is empty.")

# Calculate normalization statistics using training only
print("\nCalculating normalization statistics...")

mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True)

std[std < 1e-8] = 1.0

# Apply the same statistics to all three sets
X_train_norm = ((X_train - mean) / std).astype(np.float32)
X_val_norm = ((X_val - mean) / std).astype(np.float32)
X_test_norm = ((X_test - mean) / std).astype(np.float32)

# Create a separate output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save arrays
arrays = {
    "X_train_normalized.npy": X_train_norm,
    "X_val_normalized.npy": X_val_norm,
    "X_test_normalized.npy": X_test_norm,
    "y_train.npy": y_train,
    "y_val.npy": y_val,
    "y_test.npy": y_test,
    "normalization_mean.npy": mean,
    "normalization_std.npy": std
}

for filename, array in arrays.items():
    np.save(OUTPUT_DIR / filename, array)

# Save metadata
train_meta.to_csv(OUTPUT_DIR / "train_metadata.csv", index=False)
val_meta.to_csv(OUTPUT_DIR / "val_metadata.csv", index=False)
test_meta.to_csv(OUTPUT_DIR / "test_metadata.csv", index=False)

print("\nEXPERIMENTAL FILES SAVED")
print("Location:", OUTPUT_DIR)

for filename in arrays:
    print("[OK]", filename)

print("\nExperimental preprocessing completed successfully.")
print("Original datasets and models were not modified.")
print("=" * 65)
