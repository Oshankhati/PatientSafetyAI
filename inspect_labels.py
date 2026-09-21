from pathlib import Path
import pandas as pd
import re

DATASET_PATH = Path(r"D:\PatientSafetyAI\datasets\UP-Fall")

print("=" * 70)
print("UP-FALL LABEL AND ACTIVITY INSPECTION")
print("=" * 70)

csv_files = sorted(DATASET_PATH.rglob("*.csv"))

print(f"\nTotal CSV files: {len(csv_files)}")

# ============================================================
# 1. Inspect every file safely
# ============================================================

total_labels = {}
valid_files = []
problem_files = []

activity_files = {}
subjects = {}
cameras = {}

print("\n" + "-" * 70)
print("FILE INSPECTION")
print("-" * 70)

for file in csv_files:

    try:
        df = pd.read_csv(file)

        # --------------------------------------------
        # Check LABEL column
        # --------------------------------------------

        if "LABEL" not in df.columns:

            problem_files.append(
                (file, "LABEL column missing", list(df.columns))
            )

            print(
                f"{file.parent.name:10} | "
                f"{file.name:15} | "
                f"WARNING: LABEL missing"
            )

            continue

        # --------------------------------------------
        # Valid file
        # --------------------------------------------

        valid_files.append(file)

        counts = df["LABEL"].value_counts().to_dict()

        for label, count in counts.items():
            total_labels[label] = (
                total_labels.get(label, 0) + count
            )

        print(
            f"{file.parent.name:10} | "
            f"{file.name:15} | "
            f"frames: {len(df):3} | "
            f"labels: {counts}"
        )

        # --------------------------------------------
        # Activity
        # --------------------------------------------

        activity_match = re.search(
            r"A(\d+)",
            file.name
        )

        if activity_match:

            activity = int(
                activity_match.group(1)
            )

            activity_files[activity] = (
                activity_files.get(activity, 0) + 1
            )

        # --------------------------------------------
        # Subject
        # --------------------------------------------

        subject_match = re.search(
            r"S(\d+)",
            file.name
        )

        if subject_match:

            subject = int(
                subject_match.group(1)
            )

            subjects[subject] = (
                subjects.get(subject, 0) + 1
            )

        # --------------------------------------------
        # Camera
        # --------------------------------------------

        camera_match = re.search(
            r"C(\d+)",
            file.name
        )

        if camera_match:

            camera = int(
                camera_match.group(1)
            )

            cameras[camera] = (
                cameras.get(camera, 0) + 1
            )

    except Exception as e:

        problem_files.append(
            (file, f"Read error: {e}", [])
        )

        print(
            f"{file.parent.name:10} | "
            f"{file.name:15} | "
            f"ERROR: {e}"
        )


# ============================================================
# 2. Total label distribution
# ============================================================

print("\n" + "-" * 70)
print("TOTAL LABEL DISTRIBUTION")
print("-" * 70)

for label in sorted(total_labels):

    print(
        f"Label {label}: "
        f"{total_labels[label]} frames"
    )

total_frames = sum(total_labels.values())

print(f"\nTotal labelled frames: {total_frames}")

if total_frames > 0:

    for label in sorted(total_labels):

        percentage = (
            total_labels[label]
            / total_frames
            * 100
        )

        print(
            f"Label {label}: "
            f"{percentage:.2f}%"
        )


# ============================================================
# 3. Activity distribution
# ============================================================

print("\n" + "-" * 70)
print("ACTIVITY DISTRIBUTION")
print("-" * 70)

for activity in sorted(activity_files):

    print(
        f"A{activity}: "
        f"{activity_files[activity]} files"
    )


# ============================================================
# 4. Subject distribution
# ============================================================

print("\n" + "-" * 70)
print("SUBJECT DISTRIBUTION")
print("-" * 70)

for subject in sorted(subjects):

    print(
        f"Subject {subject}: "
        f"{subjects[subject]} valid files"
    )


# ============================================================
# 5. Camera distribution
# ============================================================

print("\n" + "-" * 70)
print("CAMERA DISTRIBUTION")
print("-" * 70)

for camera in sorted(cameras):

    print(
        f"Camera {camera}: "
        f"{cameras[camera]} valid files"
    )


# ============================================================
# 6. Problem files
# ============================================================

print("\n" + "-" * 70)
print("PROBLEM FILES")
print("-" * 70)

if not problem_files:

    print("No problem files found.")

else:

    for file, reason, columns in problem_files:

        print(f"\nFile: {file}")
        print(f"Reason: {reason}")

        if columns:
            print("Columns:")
            print(columns)


# ============================================================
# 7. Summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nTotal CSV files:       {len(csv_files)}")
print(f"Valid labelled files:  {len(valid_files)}")
print(f"Problem files:         {len(problem_files)}")
print(f"Labelled frames:       {total_frames}")

print("\nInspection complete.")
print("=" * 70)