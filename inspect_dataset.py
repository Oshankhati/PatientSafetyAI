from pathlib import Path
import pandas as pd

# ============================================================
# UP-Fall Dataset Inspection
# ============================================================

DATASET_PATH = Path(r"D:\PatientSafetyAI\datasets\UP-Fall")

print("=" * 70)
print("UP-FALL DATASET INSPECTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Check dataset folder
# ------------------------------------------------------------

if not DATASET_PATH.exists():
    print("\nERROR: Dataset folder not found!")
    print(f"Expected location: {DATASET_PATH}")
    raise SystemExit

print(f"\nDataset location:")
print(DATASET_PATH)

# ------------------------------------------------------------
# 2. Find all CSV files
# ------------------------------------------------------------

csv_files = sorted(DATASET_PATH.rglob("*.csv"))

print(f"\nTotal CSV files found: {len(csv_files)}")

# ------------------------------------------------------------
# 3. Show files by subject
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("FILES BY SUBJECT")
print("-" * 70)

for subject in sorted(DATASET_PATH.glob("SUBJECT*")):

    if subject.is_dir():
        files = list(subject.glob("*.csv"))

        print(f"{subject.name}: {len(files)} CSV files")

# ------------------------------------------------------------
# 4. Inspect the first CSV
# ------------------------------------------------------------

if len(csv_files) == 0:
    print("\nNo CSV files found.")
    raise SystemExit

first_file = csv_files[0]

print("\n" + "-" * 70)
print("FIRST FILE INSPECTION")
print("-" * 70)

print(f"\nFile:")
print(first_file)

df = pd.read_csv(first_file)

print(f"\nNumber of rows (frames): {len(df)}")
print(f"Number of columns: {len(df.columns)}")

# ------------------------------------------------------------
# 5. Show column names
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("COLUMN NAMES")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")

# ------------------------------------------------------------
# 6. Show first 5 rows
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("FIRST 5 ROWS")
print("-" * 70)

print(df.head())

# ------------------------------------------------------------
# 7. Check missing values
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("MISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()

missing_columns = missing[missing > 0]

if len(missing_columns) == 0:
    print("No missing values found in this file.")
else:
    print(missing_columns)

# ------------------------------------------------------------
# 8. Basic statistics
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("DATA TYPES")
print("-" * 70)

print(df.dtypes.value_counts())

# ------------------------------------------------------------
# 9. Inspect every CSV for frame count
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("FRAMES PER FILE")
print("-" * 70)

for file in csv_files:

    try:
        data = pd.read_csv(file)

        print(
            f"{file.parent.name:10} | "
            f"{file.name:15} | "
            f"{len(data):6} frames"
        )

    except Exception as e:

        print(
            f"{file.name:15} | ERROR: {e}"
        )

# ------------------------------------------------------------
# Finished
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)