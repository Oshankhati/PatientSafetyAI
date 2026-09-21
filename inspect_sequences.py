import pandas as pd
from pathlib import Path

print("=" * 70)
print("UP-FALL SEQUENCE METADATA INSPECTION")
print("=" * 70)

metadata_path = Path(
    r"D:\PatientSafetyAI\processed\sequences\sequence_metadata.csv"
)

if not metadata_path.exists():
    print("\nERROR: sequence_metadata.csv was not found.")
    print(metadata_path)
    exit()

df = pd.read_csv(metadata_path)

print("\n" + "-" * 70)
print("BASIC INFORMATION")
print("-" * 70)

print(f"Total sequences: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
for i, column in enumerate(df.columns, 1):
    print(f"{i:2}. {column}")

print("\n" + "-" * 70)
print("FIRST 10 SEQUENCES")
print("-" * 70)

print(df.head(10).to_string(index=False))

print("\n" + "-" * 70)
print("UNIQUE VALUES")
print("-" * 70)

for column in df.columns:
    if df[column].nunique() <= 20:
        print(f"\n{column}:")
        print(df[column].value_counts().sort_index().to_string())

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)