from pathlib import Path
import pandas as pd

# ============================================================
# UP-Fall Label Transition Analysis
# ============================================================

DATASET = Path(r"D:\PatientSafetyAI\processed\UP-Fall")

print("=" * 70)
print("UP-FALL LABEL TRANSITION ANALYSIS")
print("=" * 70)

files = sorted(DATASET.rglob("*.csv"))

total_files = len(files)

files_with_transition = 0
files_without_transition = 0

transition_counts = []

print(f"\nTotal files: {total_files}")

print("\n" + "-" * 70)
print("FILE-BY-FILE TRANSITIONS")
print("-" * 70)

for file in files:

    df = pd.read_csv(file)

    labels = df["LABEL"].tolist()

    transitions = []

    for i in range(1, len(labels)):

        if labels[i] != labels[i - 1]:

            transitions.append(
                (i - 1, i, labels[i - 1], labels[i])
            )

    if transitions:

        files_with_transition += 1

        transition_counts.append(
            len(transitions)
        )

        transition_text = "; ".join(
            f"{a}->{b}: {old}->{new}"
            for a, b, old, new in transitions
        )

        print(
            f"{file.parent.name:10s} | "
            f"{file.name:20s} | "
            f"{transition_text}"
        )

    else:

        files_without_transition += 1

        unique = sorted(set(labels))

        print(
            f"{file.parent.name:10s} | "
            f"{file.name:20s} | "
            f"NO TRANSITION | "
            f"Labels: {unique}"
        )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"\nFiles with label transition: "
    f"{files_with_transition}"
)

print(
    f"Files without transition:     "
    f"{files_without_transition}"
)

if transition_counts:

    print(
        f"\nMinimum transitions/file: "
        f"{min(transition_counts)}"
    )

    print(
        f"Maximum transitions/file: "
        f"{max(transition_counts)}"
    )

    print(
        f"Average transitions/file: "
        f"{sum(transition_counts) / len(transition_counts):.2f}"
    )

print("\n" + "=" * 70)