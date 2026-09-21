from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# UP-Fall Skeleton Inspection
# ============================================================

DATASET = Path(r"D:\PatientSafetyAI\processed\UP-Fall")

print("=" * 70)
print("UP-FALL SKELETON INSPECTION")
print("=" * 70)

# ------------------------------------------------------------
# Select first file
# ------------------------------------------------------------

files = sorted(DATASET.rglob("*.csv"))

if not files:
    raise FileNotFoundError("No CSV files found.")

file = files[0]

print("\nFile:")
print(file)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

df = pd.read_csv(file)

# ------------------------------------------------------------
# Skeleton columns
# ------------------------------------------------------------

skeleton_columns = [
    c for c in df.columns
    if c != "LABEL"
]

print("\nNumber of skeleton features:")
print(len(skeleton_columns))

# ------------------------------------------------------------
# Joint information
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("JOINT STRUCTURE")
print("-" * 70)

for joint in range(1, 34):

    x = f"Joint{joint}_X"
    y = f"Joint{joint}_Y"
    z = f"Joint{joint}_Z"

    print(
        f"Joint {joint:02d}: "
        f"{x}, {y}, {z}"
    )

# ------------------------------------------------------------
# Coordinate ranges
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("COORDINATE RANGES")
print("-" * 70)

for axis in ["X", "Y", "Z"]:

    columns = [
        f"Joint{j}_{axis}"
        for j in range(1, 34)
    ]

    values = df[columns].values

    print(
        f"{axis}: "
        f"min={values.min():.6f}, "
        f"max={values.max():.6f}, "
        f"mean={values.mean():.6f}, "
        f"std={values.std():.6f}"
    )

# ------------------------------------------------------------
# Check whether coordinates are normalized
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("NORMALIZATION CHECK")
print("-" * 70)

for axis in ["X", "Y", "Z"]:

    columns = [
        f"Joint{j}_{axis}"
        for j in range(1, 34)
    ]

    values = df[columns].values

    print(
        f"{axis}: "
        f"range = "
        f"{values.min():.4f} → "
        f"{values.max():.4f}"
    )

# ------------------------------------------------------------
# Label transitions
# ------------------------------------------------------------

labels = df["LABEL"].values

changes = np.where(
    labels[1:] != labels[:-1]
)[0]

print("\n" + "-" * 70)
print("LABEL TRANSITIONS IN FIRST FILE")
print("-" * 70)

print(
    f"Number of label transitions: {len(changes)}"
)

if len(changes) > 0:

    print("\nTransition frames:")

    for index in changes:

        print(
            f"Frame {index} → Frame {index + 1}: "
            f"{labels[index]} → {labels[index + 1]}"
        )

# ------------------------------------------------------------
# Complete
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SKELETON INSPECTION COMPLETE")
print("=" * 70)