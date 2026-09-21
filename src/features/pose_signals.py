"""Interpretable pose signals from COCO-17 keypoints (prototype, not clinical)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.pose.schemas import PersonPose

L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_ANK, R_ANK = 15, 16
NOSE = 0


def _ok(xy: np.ndarray, conf: np.ndarray, i: int, thr: float) -> bool:
    return i < len(xy) and conf[i] >= thr and not (xy[i, 0] == 0 and xy[i, 1] == 0)


def _mid(xy: np.ndarray, conf: np.ndarray, a: int, b: int, thr: float) -> np.ndarray | None:
    pts = []
    if _ok(xy, conf, a, thr):
        pts.append(xy[a])
    if _ok(xy, conf, b, thr):
        pts.append(xy[b])
    if not pts:
        return None
    return np.mean(np.stack(pts), axis=0)


@dataclass(slots=True)
class PoseSignals:
    """Scalar signals in [0, 1] where higher generally means more concern."""

    body_height: float
    torso_angle_deg: float
    bbox_aspect: float
    hip_y: float
    vertical_velocity: float
    mean_keypoint_conf: float
    zone: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def compute_pose_signals(
    person: PersonPose,
    prev_hip_y: float | None = None,
    dt: float = 1.0,
    conf_threshold: float = 0.30,
    zone: str = "unknown",
) -> PoseSignals:
    xy = person.keypoints_normalized
    conf = person.keypoint_confidence
    box = person.bounding_box
    bw = max(box.x2 - box.x1, 1.0)
    bh = max(box.y2 - box.y1, 1.0)
    aspect = float(bw / bh)  # fallen poses often wider than tall in the image

    shoulders = _mid(xy, conf, L_SHO, R_SHO, conf_threshold)
    hips = _mid(xy, conf, L_HIP, R_HIP, conf_threshold)
    ankles = _mid(xy, conf, L_ANK, R_ANK, conf_threshold)

    height = 0.0
    if shoulders is not None and ankles is not None:
        height = float(abs(ankles[1] - shoulders[1]))
    elif _ok(xy, conf, NOSE, conf_threshold) and ankles is not None:
        height = float(abs(ankles[1] - xy[NOSE, 1]))

    angle = 0.0
    if shoulders is not None and hips is not None:
        torso = shoulders - hips
        # Image y increases downward; vertical-up is (0, -1)
        norm = float(np.linalg.norm(torso)) + 1e-6
        cos = float(np.clip(np.dot(torso / norm, np.array([0.0, -1.0])), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos)))

    hip_y = float(hips[1]) if hips is not None else 0.0
    vel = 0.0
    if prev_hip_y is not None and dt > 0:
        vel = (hip_y - prev_hip_y) / dt  # positive = moving down the image

    return PoseSignals(
        body_height=height,
        torso_angle_deg=angle,
        bbox_aspect=aspect,
        hip_y=hip_y,
        vertical_velocity=vel,
        mean_keypoint_conf=person.mean_keypoint_confidence(),
        zone=zone,
    )
