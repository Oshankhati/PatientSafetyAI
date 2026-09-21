import numpy as np
from pathlib import Path

print("=" * 70)
print("UP-FALL DATA NORMALIZATION")
print("=" * 70)

BASE_DIR = Path(r"D:\PatientSafetyAI\processed\sequences\split")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

X_train = np.load(BASE_DIR / "X_train.npy")
X_test = np.load(BASE_DIR / "X_test.npy")

y_train = np.load(BASE_DIR / "y_train.npy")
y_test = np.load(BASE_DIR / "y_test.npy")

print("\nOriginal shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

# ------------------------------------------------------------
# CALCULATE TRAINING STATISTICS
# ------------------------------------------------------------

# Calculate mean and standard deviation using ONLY training data.
mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True)

# Prevent division by zero.
std[std == 0] = 1.0

# ------------------------------------------------------------
# NORMALIZE
# ------------------------------------------------------------

X_train_normalized = (X_train - mean) / std
X_test_normalized = (X_test - mean) / std

# ------------------------------------------------------------
# CHECK RESULTS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("NORMALIZATION CHECK")
print("-" * 70)

print("\nTraining data:")
print(f"Mean before: {X_train.mean():.6f}")
print(f"Std before:  {X_train.std():.6f}")

print("\nTraining data after normalization:")
print(f"Mean: {X_train_normalized.mean():.6f}")
print(f"Std:  {X_train_normalized.std():.6f}")

print("\nTest data after normalization:")
print(f"Mean: {X_test_normalized.mean():.6f}")
print(f"Std:  {X_test_normalized.std():.6f}")

# ------------------------------------------------------------
# SAVE NORMALIZED DATA
# ------------------------------------------------------------

np.save(BASE_DIR / "X_train_normalized.npy", X_train_normalized)
np.save(BASE_DIR / "X_test_normalized.npy", X_test_normalized)

# Save the normalization parameters.
np.save(BASE_DIR / "normalization_mean.npy", mean)
np.save(BASE_DIR / "normalization_std.npy", std)

# Labels don't need normalization.
np.save(BASE_DIR / "y_train.npy", y_train)
np.save(BASE_DIR / "y_test.npy", y_test)

print("\n" + "-" * 70)
print("FILES SAVED")
print("-" * 70)

print("[OK] X_train_normalized.npy")
print("[OK] X_test_normalized.npy")
print("[OK] normalization_mean.npy")
print("[OK] normalization_std.npy")

print("\n" + "=" * 70)
print("NORMALIZATION COMPLETE")
print("=" * 70)