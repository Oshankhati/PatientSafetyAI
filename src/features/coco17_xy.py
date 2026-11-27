"""Deployment-shaped COCO-17 XY features and UP-Fall joint projection."""

from __future__ import annotations

import numpy as np

from src.pose.schemas import PersonPose

# These are the direct 1:1 joints named in yolo_to_upfall.py's documented
# UP-Fall joint table, expressed as zero-based indices into its 33-joint array.
UPFALL_COCO17_JOINTS: tuple[int, ...] = (
    27,  # nose
    28,  # left eye
    30,  # right eye
    29,  # left ear
    31,  # right ear
    5,   # left shoulder
    12,  # right shoulder
    6,   # left elbow
    13,  # right elbow
    7,   # left wrist
    14,  # right wrist
    18,  # left hip
    22,  # right hip
    19,  # left knee
    23,  # right knee
    20,  # left ankle
    24,  # right ankle
)

COCO17_XY_FEATURE_DIM = 17 * 2


def upfall99_to_coco17_xy(sequences: np.ndarray) -> np.ndarray:
    """Project UP-Fall ``(..., 99)`` XYZ data to the corresponding COCO-17 XY joints."""
    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != 99:
        raise ValueError(f"Expected (..., 99) UP-Fall features, got {values.shape}")
    joints = values.reshape(*values.shape[:-1], 33, 3)
    return joints[..., UPFALL_COCO17_JOINTS, :2].reshape(
        *values.shape[:-1], COCO17_XY_FEATURE_DIM
    )


def person_to_coco17_xy(person: PersonPose) -> np.ndarray:
    """Flatten normalized COCO-17 XY coordinates without synthesizing depth."""
    xy = np.asarray(person.keypoints_normalized, dtype=np.float32)
    if xy.shape != (17, 2):
        raise ValueError(f"Expected keypoints_normalized shape (17, 2), got {xy.shape}")
    return xy.reshape(COCO17_XY_FEATURE_DIM)
