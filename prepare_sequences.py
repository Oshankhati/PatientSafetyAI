import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# UP-FALL SEQUENCE PREPARATION
# ============================================================

DATASET_DIR = Path(r"D:\PatientSafetyAI\processed\UP-Fall")
OUTPUT_DIR = Path(r"D:\PatientSafetyAI\processed\sequences")

# Number of consecutive frames used as one sample
WINDOW_SIZE = 20

# Step size between consecutive windows
STEP_SIZE = 5

# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("UP-FALL SEQUENCE PREPARATION")
print("=" * 70)

print(f"\nDataset: {DATASET_DIR}")
print(f"Window size: {WINDOW_SIZE} frames")
print(f"Step size: {STEP_SIZE} frames")

# ------------------------------------------------------------
# Find all CSV files
# ------------------------------------------------------------

csv_files = sorted(DATASET_DIR.rglob("*.csv"))

print(f"\nCSV recordings found: {len(csv_files)}")

# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

X_sequences = []
y_sequences = []

metadata = []

# ------------------------------------------------------------
# Process each recording
# ------------------------------------------------------------

for file_path in csv_files:

    try:
        df = pd.read_csv(file_path)

        # Make sure LABEL exists
        if "LABEL" not in df.columns:
            print(f"[SKIP] LABEL missing: {file_path.name}")
            continue

        # ----------------------------------------------------
        # Extract features and labels
        # ----------------------------------------------------

        feature_columns = [
            col for col in df.columns
            if col != "LABEL"
        ]

        X = df[feature_columns].values.astype(np.float32)
        labels = df["LABEL"].values.astype(np.int64)

        # ----------------------------------------------------
        # Create temporal windows
        # ----------------------------------------------------

        for start in range(
            0,
            len(X) - WINDOW_SIZE + 1,
            STEP_SIZE
        ):

            end = start + WINDOW_SIZE

            sequence = X[start:end]

            # Use the label of the final frame.
            #
            # This represents the state the model should
            # recognize at the end of the observed sequence.
            target = labels[end - 1]

            X_sequences.append(sequence)
            y_sequences.append(target)

            metadata.append({
                "subject": file_path.parent.name,
                "file": file_path.name,
                "start_frame": start,
                "end_frame": end - 1,
                "label": target
            })

        print(
            f"[OK] {file_path.parent.name}\\{file_path.name}"
            f" -> {len(X) - WINDOW_SIZE + 1} possible frames"
        )

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")

# ------------------------------------------------------------
# Convert to NumPy arrays
# ------------------------------------------------------------

X_sequences = np.array(X_sequences, dtype=np.float32)
y_sequences = np.array(y_sequences, dtype=np.int64)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

np.save(
    OUTPUT_DIR / "X_sequences.npy",
    X_sequences
)

np.save(
    OUTPUT_DIR / "y_sequences.npy",
    y_sequences
)

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(
    OUTPUT_DIR / "sequence_metadata.csv",
    index=False
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SEQUENCE PREPARATION COMPLETE")
print("=" * 70)

print(f"\nTotal sequences: {len(X_sequences)}")
print(f"Sequence shape: {X_sequences.shape}")
print(f"Labels shape:   {y_sequences.shape}")

if len(y_sequences) > 0:

    unique, counts = np.unique(
        y_sequences,
        return_counts=True
    )

    print("\nSEQUENCE LABEL DISTRIBUTION")

    for label, count in zip(unique, counts):

        percentage = (
            count / len(y_sequences)
        ) * 100

        print(
            f"Label {label}: "
            f"{count} sequences "
            f"({percentage:.2f}%)"
        )

print("\nExpected input format for deep learning:")
print(
    "(samples, time_steps, features)"
)

print(
    f"({len(X_sequences)}, "
    f"{WINDOW_SIZE}, "
    f"99)"
)

print("\nSaved files:")

print(
    OUTPUT_DIR / "X_sequences.npy"
)

print(
    OUTPUT_DIR / "y_sequences.npy"
)

print(
    OUTPUT_DIR / "sequence_metadata.csv"
)

print("\n" + "=" * 70)