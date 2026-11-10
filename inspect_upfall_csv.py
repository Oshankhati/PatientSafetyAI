
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_DIR = Path(
    r"C:\Projects\PatientSafetyAI\datasets\processed\UP-Fall"
)

csv_files = sorted(DATASET_DIR.rglob("*.csv"))

print("CSV FILES FOUND:", len(csv_files))

# Inspect one recording from each subject
for subject in range(1, 6):

    matching = [
        f for f in csv_files
        if f.parent.name.upper() == f"SUBJECT{subject}"
    ]

    if not matching:
        print(f"\nSUBJECT{subject}: No recording found")
        continue

    file_path = matching[0]
    df = pd.read_csv(file_path)

    feature_columns = [
        col for col in df.columns
        if col != "LABEL"
    ]

    X = df[feature_columns].to_numpy(dtype=np.float32)

    print("\n" + "=" * 65)
    print("SUBJECT:", subject)
    print("FILE:", file_path.name)
    print("FEATURE COUNT:", len(feature_columns))
    print("DATA SHAPE:", X.shape)

    print("\nFIRST 15 FEATURE COLUMNS:")
    print(feature_columns[:15])

    print("\nFIRST 9 FEATURE VALUES:")
    print(X[0, :9])

    print("\nFEATURE STATISTICS:")
    print("Mean:", np.mean(X))
    print("Std:", np.std(X))
    print("Minimum:", np.min(X))
    print("Maximum:", np.max(X))

    print("\nZERO PERCENTAGE:", np.mean(X == 0) * 100)

print("\nINSPECTION COMPLETED")
