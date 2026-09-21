from pathlib import Path
import pandas as pd

# ============================================================
# UP-Fall Cleaned Dataset Verification
# ============================================================

DATASET = Path(r"D:\PatientSafetyAI\processed\UP-Fall")

print("=" * 70)
print("UP-FALL CLEANED DATASET VERIFICATION")
print("=" * 70)

csv_files = sorted(DATASET.rglob("*.csv"))

print(f"\nCSV files found: {len(csv_files)}")

expected_features = 99
expected_columns = 100
expected_frames = 100

total_frames = 0
total_label_0 = 0
total_label_1 = 0

problems = []

# ------------------------------------------------------------
# Check every file
# ------------------------------------------------------------

for file in csv_files:

    try:

        df = pd.read_csv(file)

        # Check columns
        if len(df.columns) != expected_columns:

            problems.append(
                f"{file.name}: "
                f"{len(df.columns)} columns"
            )

        # Check frames
        if len(df) != expected_frames:

            problems.append(
                f"{file.name}: "
                f"{len(df)} frames"
            )

        # Check LABEL
        if "LABEL" not in df.columns:

            problems.append(
                f"{file.name}: LABEL missing"
            )
            continue

        # Check feature count
        feature_columns = [
            c for c in df.columns
            if c != "LABEL"
        ]

        if len(feature_columns) != expected_features:

            problems.append(
                f"{file.name}: "
                f"{len(feature_columns)} features"
            )

        # Check missing values
        missing = df.isnull().sum().sum()

        if missing > 0:

            problems.append(
                f"{file.name}: "
                f"{missing} missing values"
            )

        # Check labels
        unique_labels = set(
            df["LABEL"].unique()
        )

        if not unique_labels.issubset({0, 1}):

            problems.append(
                f"{file.name}: "
                f"unexpected labels {unique_labels}"
            )

        # Count
        counts = df["LABEL"].value_counts()

        total_label_0 += counts.get(0, 0)
        total_label_1 += counts.get(1, 0)

        total_frames += len(df)

    except Exception as e:

        problems.append(
            f"{file.name}: {e}"
        )


# ============================================================
# Results
# ============================================================

print("\n" + "-" * 70)
print("VERIFICATION RESULTS")
print("-" * 70)

print(
    f"\nFiles:              {len(csv_files)}"
)

print(
    f"Total frames:       {total_frames}"
)

print(
    f"Total features:     {expected_features} per frame"
)

print(
    f"Columns:            {expected_columns} per file"
)

print(
    f"Label 0:            {total_label_0}"
)

print(
    f"Label 1:            {total_label_1}"
)

print(
    f"Missing values:     checked"
)

print(
    f"Label values:       checked"
)

print("\n" + "-" * 70)

if len(problems) == 0:

    print("STATUS: DATASET IS CLEAN")
    print("All files passed verification.")

else:

    print(
        f"STATUS: {len(problems)} PROBLEM(S) FOUND"
    )

    print("\nProblems:")

    for problem in problems:

        print(
            f"  - {problem}"
        )

print("\n" + "=" * 70)