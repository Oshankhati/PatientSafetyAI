from pathlib import Path
import pandas as pd

# ============================================================
# UP-Fall Dataset Cleaner
# ============================================================

SOURCE = Path(r"D:\PatientSafetyAI\datasets\UP-Fall")
DESTINATION = Path(r"D:\PatientSafetyAI\processed\UP-Fall")

print("=" * 70)
print("UP-FALL DATASET CLEANING")
print("=" * 70)

# ------------------------------------------------------------
# Create destination
# ------------------------------------------------------------

DESTINATION.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Find CSV files
# ------------------------------------------------------------

csv_files = sorted(SOURCE.rglob("*.csv"))

print(f"\nSource CSV files found: {len(csv_files)}")
print(f"Destination: {DESTINATION}")

# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

processed = 0
failed = 0

label_renamed = 0

total_frames = 0

label_counts = {
    0: 0,
    1: 0
}

# ------------------------------------------------------------
# Process every CSV
# ------------------------------------------------------------

for source_file in csv_files:

    try:

        # Read original file
        df = pd.read_csv(source_file)

        # ----------------------------------------------------
        # Check number of rows
        # ----------------------------------------------------

        if len(df) != 100:

            print(
                f"\nWARNING: {source_file.name} "
                f"has {len(df)} frames instead of 100."
            )

        # ----------------------------------------------------
        # Identify label column
        # ----------------------------------------------------

        if "LABEL" in df.columns:

            label_column = "LABEL"

        elif "LLABEL" in df.columns:

            label_column = "LLABEL"

            df = df.rename(
                columns={"LLABEL": "LABEL"}
            )

            label_renamed += 1

        elif len(df.columns) == 100:

            # The problematic file has the final
            # column named "0".
            #
            # Since the first 99 columns are the
            # expected skeleton coordinates, the
            # final column is the label column.

            label_column = df.columns[-1]

            df = df.rename(
                columns={label_column: "LABEL"}
            )

            label_renamed += 1

        else:

            raise ValueError(
                "Could not identify LABEL column"
            )

        # ----------------------------------------------------
        # Check feature count
        # ----------------------------------------------------

        feature_columns = [
            column
            for column in df.columns
            if column != "LABEL"
        ]

        if len(feature_columns) != 99:

            raise ValueError(
                f"Expected 99 skeleton features, "
                f"found {len(feature_columns)}"
            )

        # ----------------------------------------------------
        # Check labels
        # ----------------------------------------------------

        unique_labels = set(
            df["LABEL"].dropna().unique()
        )

        if not unique_labels.issubset({0, 1}):

            raise ValueError(
                f"Unexpected label values: "
                f"{unique_labels}"
            )

        # ----------------------------------------------------
        # Count labels
        # ----------------------------------------------------

        counts = df["LABEL"].value_counts().to_dict()

        label_counts[0] += counts.get(0, 0)
        label_counts[1] += counts.get(1, 0)

        total_frames += len(df)

        # ----------------------------------------------------
        # Create destination subject folder
        # ----------------------------------------------------

        relative_path = source_file.relative_to(SOURCE)

        destination_file = (
            DESTINATION / relative_path
        )

        destination_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save cleaned file
        # ----------------------------------------------------

        df.to_csv(
            destination_file,
            index=False
        )

        processed += 1

        print(
            f"[OK] {relative_path}"
        )

    except Exception as error:

        failed += 1

        print(
            f"[FAILED] {source_file}"
        )

        print(
            f"         Reason: {error}"
        )


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(
    f"\nFiles found:          {len(csv_files)}"
)

print(
    f"Files processed:      {processed}"
)

print(
    f"Files failed:         {failed}"
)

print(
    f"Label columns fixed:  {label_renamed}"
)

print(
    f"Total frames:         {total_frames}"
)

print("\nLABEL DISTRIBUTION")

print(
    f"Label 0: {label_counts[0]}"
)

print(
    f"Label 1: {label_counts[1]}"
)

if total_frames > 0:

    print(
        f"\nLabel 0: "
        f"{label_counts[0] / total_frames * 100:.2f}%"
    )

    print(
        f"Label 1: "
        f"{label_counts[1] / total_frames * 100:.2f}%"
    )

print(
    f"\nCleaned dataset saved to:"
)

print(DESTINATION)

print("\n" + "=" * 70)