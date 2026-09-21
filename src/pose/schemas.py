"""Pose data schemas and COCO keypoint constants.

Feature-format note
-------------------
UP-Fall LSTM baseline expects **33 Kinect-style joints × (X, Y, Z) = 99**.
YOLO Pose (COCO) produces **17 keypoints × (x, y [, conf])**.

These representations are **not interchangeable**. Live video uses the
explicit approximate mapping in ``src/features/yolo_to_upfall.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# COCO-17 keypoint names used by Ultralytics YOLO Pose
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Skeleton edges for visualization (pairs of COCO indices)
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# UP-Fall / Kinect-style joint count used by the existing LSTM
UPFALL_JOINT_COUNT = 33
UPFALL_FEATURE_DIM = UPFALL_JOINT_COUNT * 3  # XYZ


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned person box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0

    @property
    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class PersonPose:
    """Pose of a single detected person in one frame.

    ``keypoints`` shape: (17, 2) — pixel (x, y)
    ``keypoint_confidence`` shape: (17,)
    ``keypoints_normalized`` shape: (17, 2) — coords in [0, 1] relative to frame
    """

    person_id: int
    bounding_box: BoundingBox
    keypoints: np.ndarray
    keypoint_confidence: np.ndarray
    keypoints_normalized: np.ndarray
    frame_number: int
    timestamp: float
    detection_confidence: float = 0.0
    format: str = "coco17"

    def mean_keypoint_confidence(self) -> float:
        conf = self.keypoint_confidence
        if conf is None or len(conf) == 0:
            return 0.0
        return float(np.nanmean(conf))

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "bounding_box": self.bounding_box.to_dict(),
            "keypoints": self.keypoints.tolist(),
            "keypoint_confidence": self.keypoint_confidence.tolist(),
            "keypoints_normalized": self.keypoints_normalized.tolist(),
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "detection_confidence": self.detection_confidence,
            "format": self.format,
            "mean_keypoint_confidence": self.mean_keypoint_confidence(),
        }


@dataclass(slots=True)
class FramePoseResult:
    """All person poses detected in a single video frame."""

    frame_number: int
    timestamp: float
    persons: list[PersonPose] = field(default_factory=list)
    fps_hint: float | None = None

    @property
    def person_count(self) -> int:
        return len(self.persons)

    def select_person(
        self,
        person_id: int | None = None,
        strategy: str = "largest_box",
    ) -> PersonPose | None:
        """Select one person for monitoring.

        Parameters
        ----------
        person_id:
            Prefer this track/detection ID when present.
        strategy:
            Fallback when ``person_id`` is None or missing:
            ``largest_box`` (default) or ``highest_confidence``.
        """
        if not self.persons:
            return None
        if person_id is not None:
            for person in self.persons:
                if person.person_id == person_id:
                    return person
        if strategy == "highest_confidence":
            return max(self.persons, key=lambda p: p.detection_confidence)
        # largest bounding box area
        return max(
            self.persons,
            key=lambda p: (p.bounding_box.x2 - p.bounding_box.x1)
            * (p.bounding_box.y2 - p.bounding_box.y1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "person_count": self.person_count,
            "persons": [p.to_dict() for p in self.persons],
            "fps_hint": self.fps_hint,
        }
