from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# UP-Fall Movement Analysis Around Label Transition
# ============================================================

DATASET = Path(r"D:\PatientSafetyAI\processed\UP-Fall")

print("=" * 70)
print("UP-FALL MOVEMENT ANALYSIS")
print("=" * 70)

files = sorted(DATASET.rglob("*.csv"))

all_results = []

# ------------------------------------------------------------
# Analyze every recording
# ------------------------------------------------------------

for file in files:

    df = pd.read_csv(file)

    # --------------------------------------------------------
    # Find transition
    # --------------------------------------------------------

    labels = df["LABEL"].values

    transition_indices = np.where(
        labels[1:] != labels[:-1]
    )[0]

    if len(transition_indices) != 1:
        continue

    transition = transition_indices[0] + 1

    # --------------------------------------------------------
    # Extract coordinates
    # Shape:
    # frames × joints × coordinates
    # --------------------------------------------------------

    coordinates = []

    for joint in range(1, 34):

        x = df[f"Joint{joint}_X"].values
        y = df[f"Joint{joint}_Y"].values
        z = df[f"Joint{joint}_Z"].values

        coordinates.append(
            np.column_stack((x, y, z))
        )

    coordinates = np.stack(
        coordinates,
        axis=1
    )

    # --------------------------------------------------------
    # Frame-to-frame displacement
    # --------------------------------------------------------

    displacement = np.linalg.norm(
        coordinates[1:] - coordinates[:-1],
        axis=2
    )

    # Average movement across 33 joints
    mean_movement = displacement.mean(axis=1)

    # Maximum joint movement
    max_movement = displacement.max(axis=1)

    # --------------------------------------------------------
    # Print local movement around transition
    # --------------------------------------------------------

    start = max(0, transition - 10)
    end = min(len(mean_movement), transition + 10)

    print("\n" + "-" * 70)

    print(
        f"{file.parent.name} | "
        f"{file.name}"
    )

    print(
        f"Transition: frame "
        f"{transition - 1} → {transition}"
    )

    print("\nFrame | Label | Mean movement | Max movement")

    for frame in range(start, end):

        print(
            f"{frame:5d} | "
            f"{labels[frame]:5d} | "
            f"{mean_movement[frame]:13.6f} | "
            f"{max_movement[frame]:12.6f}"
        )

    # --------------------------------------------------------
    # Store transition movement
    # --------------------------------------------------------

    transition_movement = mean_movement[
        min(transition - 1, len(mean_movement) - 1)
    ]

    before_start = max(
        0,
        transition - 10
    )

    before_end = max(
        before_start + 1,
        transition - 1
    )

    after_start = min(
        transition,
        len(mean_movement) - 1
    )

    after_end = min(
        len(mean_movement),
        transition + 9
    )

    before_mean = mean_movement[
        before_start:before_end
    ].mean()

    after_mean = mean_movement[
        after_start:after_end
    ].mean()

    all_results.append({
        "file": file.name,
        "subject": file.parent.name,
        "transition": transition,
        "before_movement": before_mean,
        "transition_movement": transition_movement,
        "after_movement": after_mean
    })


# ============================================================
# Overall summary
# ============================================================

results = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("OVERALL MOVEMENT SUMMARY")
print("=" * 70)

if not results.empty:

    print(
        f"\nRecordings analyzed: "
        f"{len(results)}"
    )

    print(
        f"\nAverage movement BEFORE transition: "
        f"{results['before_movement'].mean():.6f}"
    )

    print(
        f"Average movement AT transition: "
        f"{results['transition_movement'].mean():.6f}"
    )

    print(
        f"Average movement AFTER transition: "
        f"{results['after_movement'].mean():.6f}"
    )

    print("\nMovement ratios:")

    before = results["before_movement"].mean()
    transition = results["transition_movement"].mean()
    after = results["after_movement"].mean()

    if before > 0:

        print(
            f"Transition / Before: "
            f"{transition / before:.2f}x"
        )

        print(
            f"After / Before: "
            f"{after / before:.2f}x"
        )

print("\n" + "=" * 70)
print("MOVEMENT ANALYSIS COMPLETE")
print("=" * 70)