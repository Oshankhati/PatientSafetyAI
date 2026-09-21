"""Extract structured keypoints from Ultralytics YOLO Pose results."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.pose.schemas import BoundingBox, FramePoseResult, PersonPose

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float32)


class KeypointExtractor:
    """Convert Ultralytics ``Results`` objects into ``FramePoseResult``."""

    def __init__(self, keypoint_conf_threshold: float = 0.30) -> None:
        self.keypoint_conf_threshold = keypoint_conf_threshold

    def extract(
        self,
        result: Any,
        frame_number: int,
        timestamp: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> FramePoseResult:
        """Parse one Ultralytics result into typed person poses."""
        persons: list[PersonPose] = []

        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)

        if boxes is None or len(boxes) == 0:
            return FramePoseResult(
                frame_number=frame_number,
                timestamp=timestamp,
                persons=[],
            )

        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        ids = None
        if getattr(boxes, "id", None) is not None:
            ids = _to_numpy(boxes.id)

        kpt_xy = None
        kpt_conf = None
        kpt_xyn = None
        if keypoints is not None and getattr(keypoints, "xy", None) is not None:
            kpt_xy = _to_numpy(keypoints.xy)
            if getattr(keypoints, "conf", None) is not None:
                kpt_conf = _to_numpy(keypoints.conf)
            if getattr(keypoints, "xyn", None) is not None:
                kpt_xyn = _to_numpy(keypoints.xyn)

        height = width = None
        if frame_shape is not None and len(frame_shape) >= 2:
            height, width = int(frame_shape[0]), int(frame_shape[1])

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i][:4])
            det_conf = float(confs[i]) if confs is not None and len(confs) > i else 0.0
            person_id = int(ids[i]) if ids is not None and len(ids) > i else i

            if kpt_xy is not None and i < len(kpt_xy):
                kxy = kpt_xy[i]
                kconf = (
                    kpt_conf[i]
                    if kpt_conf is not None and i < len(kpt_conf)
                    else np.ones(len(kxy), dtype=np.float32)
                )
                if kpt_xyn is not None and i < len(kpt_xyn):
                    kxyn = kpt_xyn[i]
                elif width and height:
                    kxyn = np.column_stack(
                        [
                            kxy[:, 0] / max(width, 1),
                            kxy[:, 1] / max(height, 1),
                        ]
                    ).astype(np.float32)
                else:
                    kxyn = kxy.copy()
            else:
                kxy = np.zeros((17, 2), dtype=np.float32)
                kconf = np.zeros((17,), dtype=np.float32)
                kxyn = np.zeros((17, 2), dtype=np.float32)

            persons.append(
                PersonPose(
                    person_id=person_id,
                    bounding_box=BoundingBox(x1, y1, x2, y2, confidence=det_conf),
                    keypoints=kxy.astype(np.float32),
                    keypoint_confidence=kconf.astype(np.float32),
                    keypoints_normalized=kxyn.astype(np.float32),
                    frame_number=frame_number,
                    timestamp=timestamp,
                    detection_confidence=det_conf,
                    format="coco17",
                )
            )

        return FramePoseResult(
            frame_number=frame_number,
            timestamp=timestamp,
            persons=persons,
        )

    def filter_low_confidence_joints(
        self,
        person: PersonPose,
        threshold: float | None = None,
    ) -> PersonPose:
        """Zero-out joints below the confidence threshold (returns a shallow copy)."""
        thr = (
            self.keypoint_conf_threshold if threshold is None else float(threshold)
        )
        mask = person.keypoint_confidence >= thr
        kxy = person.keypoints.copy()
        kxyn = person.keypoints_normalized.copy()
        kxy[~mask] = 0.0
        kxyn[~mask] = 0.0
        return PersonPose(
            person_id=person.person_id,
            bounding_box=person.bounding_box,
            keypoints=kxy,
            keypoint_confidence=person.keypoint_confidence,
            keypoints_normalized=kxyn,
            frame_number=person.frame_number,
            timestamp=person.timestamp,
            detection_confidence=person.detection_confidence,
            format=person.format,
        )
