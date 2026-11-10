import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("Datasets/processed/UP-Fall")
META_PATH = Path("Datasets/processed/sequences/sequence_metadata.csv")

metadata = pd.read_csv(META_PATH)

mixed = 0
disagreement = 0
total = 0

for _, row in metadata.iterrows():

    file_path = DATA_DIR / row["subject"] / row["file"]

    df = pd.read_csv(file_path)

    start = int(row["start_frame"])
    end = int(row["end_frame"])

    labels = df["LABEL"].iloc[start:end + 1].to_numpy()

    if len(np.unique(labels)) > 1:
        mixed += 1

    majority_label = np.bincount(labels).argmax()

    if labels[-1] != majority_label:
        disagreement += 1

    total += 1

print("\n" + "=" * 50)
print("LABEL TRANSITION ANALYSIS")
print("=" * 50)

print("Total windows:", total)
print("Mixed-label windows:", mixed)
print("Final label differs from majority:", disagreement)

print("Mixed-label percentage:", round(mixed / total * 100, 2))
print("Majority disagreement percentage:", round(disagreement / total * 100, 2))