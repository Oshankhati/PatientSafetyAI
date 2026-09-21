"""Explicit YOLO COCO-17 → UP-Fall 33-joint XYZ conversion.

This mapping is an **approximation** for wiring live pose into the existing
99-dim LSTM. It is **not** a claim that YOLO keypoints equal Kinect training
data.

Documented limitations
----------------------
- YOLO has 17 joints; UP-Fall uses 33 (Kinect-style) XYZ channels.
- YOLO has no depth; **Z is always 0**.
- Unmapped / finger joints are filled by nearest mapped joint or zeros.
- Coordinates use **frame-normalized x, y in [0, 1]** (``keypoints_normalized``).
- Domain shift remains: do not treat live LSTM scores as equivalent to the
  subject-wise UP-Fall test metrics.

Joint table (1-based UP-Fall ``JointN_*`` columns)
-------------------------------------------------
Index  Name (Azure-Kinect inspired)     Source COCO joint(s)
1      pelvis / mid-hip                 mean(left_hip, right_hip)
2      spine navel                      0.65*pelvis + 0.35*neck
3      spine chest                      0.35*pelvis + 0.65*neck
4      neck                             mean(left_shoulder, right_shoulder)
5      clavicle left                    left_shoulder
6      shoulder left                    left_shoulder
7      elbow left                       left_elbow
8      wrist left                       left_wrist
9      hand left                        left_wrist
10     hand tip left                    left_wrist
11     thumb left                       left_wrist
12     clavicle right                   right_shoulder
13     shoulder right                   right_shoulder
14     elbow right                      right_elbow
15     wrist right                      right_wrist
16     hand right                       right_wrist
17     hand tip right                   right_wrist
18     thumb right                      right_wrist
19     hip left                         left_hip
20     knee left                        left_knee
21     ankle left                       left_ankle
22     foot left                        left_ankle
23     hip right                        right_hip
24     knee right                       right_knee
25     ankle right                      right_ankle
26     foot right                       right_ankle
27     head                             mean(left_eye, right_eye) else nose
28     nose                             nose
29     eye left                         left_eye
30     ear left                         left_ear
31     eye right                        right_eye
32     ear right                        right_ear
33     head top                         nose (placeholder)

Layout of the 99-vector: ``[J1x, J1y, J1z, J2x, J2y, J2z, ..., J33z]``.
"""

from __future__ import annotations

import logging

import numpy as np

from src.pose.schemas import UPFALL_FEATURE_DIM, UPFALL_JOINT_COUNT, PersonPose

logger = logging.getLogger(__name__)

# COCO-17 indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

# Per UP-Fall joint (0-based): either a COCO index, or a derivation key.
_DERIVE_PELVIS = "pelvis"
_DERIVE_NECK = "neck"
_DERIVE_SPINE_NAVEL = "spine_navel"
_DERIVE_SPINE_CHEST = "spine_chest"
_DERIVE_HEAD = "head"

# Length 33: COCO index (int) or derivation name (str)
UPFALL_JOINT_SOURCES: tuple[int | str, ...] = (
    _DERIVE_PELVIS,  # 1
    _DERIVE_SPINE_NAVEL,  # 2
    _DERIVE_SPINE_CHEST,  # 3
    _DERIVE_NECK,  # 4
    L_SHO,  # 5
    L_SHO,  # 6
    L_ELB,  # 7
    L_WRI,  # 8
    L_WRI,  # 9
    L_WRI,  # 10
    L_WRI,  # 11
    R_SHO,  # 12
    R_SHO,  # 13
    R_ELB,  # 14
    R_WRI,  # 15
    R_WRI,  # 16
    R_WRI,  # 17
    R_WRI,  # 18
    L_HIP,  # 19
    L_KNE,  # 20
    L_ANK,  # 21
    L_ANK,  # 22
    R_HIP,  # 23
    R_KNE,  # 24
    R_ANK,  # 25
    R_ANK,  # 26
    _DERIVE_HEAD,  # 27
    NOSE,  # 28
    L_EYE,  # 29
    L_EAR,  # 30
    R_EYE,  # 31
    R_EAR,  # 32
    NOSE,  # 33 placeholder
)

assert len(UPFALL_JOINT_SOURCES) == UPFALL_JOINT_COUNT


def _valid(xy: np.ndarray, conf: np.ndarray, idx: int, thr: float) -> bool:
    if idx < 0 or idx >= len(xy):
        return False
    if conf[idx] < thr:
        return False
    return not (xy[idx, 0] == 0.0 and xy[idx, 1] == 0.0)


def _mean_points(
    xy: np.ndarray,
    conf: np.ndarray,
    indices: tuple[int, ...],
    thr: float,
) -> np.ndarray | None:
    pts = [xy[i] for i in indices if _valid(xy, conf, i, thr)]
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def coco17_to_upfall99(
    keypoints_xy: np.ndarray,
    keypoint_confidence: np.ndarray | None = None,
    conf_threshold: float = 0.30,
) -> np.ndarray:
    """Convert one person COCO-17 xy into a 99-dim UP-Fall-shaped vector.

    Parameters
    ----------
    keypoints_xy:
        Shape (17, 2), preferably **normalized** image coordinates.
    """
    xy = np.asarray(keypoints_xy, dtype=np.float32).reshape(17, 2)
    if keypoint_confidence is None:
        conf = np.ones((17,), dtype=np.float32)
    else:
        conf = np.asarray(keypoint_confidence, dtype=np.float32).reshape(17)

    pelvis = _mean_points(xy, conf, (L_HIP, R_HIP), conf_threshold)
    neck = _mean_points(xy, conf, (L_SHO, R_SHO), conf_threshold)
    head = _mean_points(xy, conf, (L_EYE, R_EYE), conf_threshold)
    if head is None and _valid(xy, conf, NOSE, conf_threshold):
        head = xy[NOSE]

    derived: dict[str, np.ndarray | None] = {
        _DERIVE_PELVIS: pelvis,
        _DERIVE_NECK: neck,
        _DERIVE_HEAD: head,
        _DERIVE_SPINE_NAVEL: None,
        _DERIVE_SPINE_CHEST: None,
    }
    if pelvis is not None and neck is not None:
        derived[_DERIVE_SPINE_NAVEL] = 0.65 * pelvis + 0.35 * neck
        derived[_DERIVE_SPINE_CHEST] = 0.35 * pelvis + 0.65 * neck

    out = np.zeros((UPFALL_JOINT_COUNT, 3), dtype=np.float32)
    for j, source in enumerate(UPFALL_JOINT_SOURCES):
        if isinstance(source, str):
            pt = derived.get(source)
            if pt is None:
                continue
            out[j, 0] = pt[0]
            out[j, 1] = pt[1]
            out[j, 2] = 0.0
            continue
        if _valid(xy, conf, source, conf_threshold):
            out[j, 0] = xy[source, 0]
            out[j, 1] = xy[source, 1]
            out[j, 2] = 0.0

    flat = out.reshape(-1)
    if flat.shape[0] != UPFALL_FEATURE_DIM:
        raise ValueError(
            f"Converted feature width {flat.shape[0]} != {UPFALL_FEATURE_DIM}"
        )
    return flat


def person_to_upfall99(
    person: PersonPose,
    conf_threshold: float = 0.30,
    use_normalized: bool = True,
) -> np.ndarray:
    """Convert a ``PersonPose`` to the UP-Fall 99-dim layout."""
    xy = person.keypoints_normalized if use_normalized else person.keypoints
    return coco17_to_upfall99(
        xy,
        person.keypoint_confidence,
        conf_threshold=conf_threshold,
    )


class YoloToUpfallConverter:
    """Explicit, documented converter (approximate; Z=0; domain shift remains)."""

    def __init__(self, conf_threshold: float = 0.30, use_normalized: bool = True) -> None:
        self.conf_threshold = conf_threshold
        self.use_normalized = use_normalized
        logger.info(
            "YOLO->UP-Fall converter ready (approx. mapping, Z=0, dim=%s)",
            UPFALL_FEATURE_DIM,
        )

    def convert(self, yolo_keypoints_xy: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
        return coco17_to_upfall99(
            yolo_keypoints_xy,
            confidence,
            conf_threshold=self.conf_threshold,
        )

    def convert_person(self, person: PersonPose) -> np.ndarray:
        return person_to_upfall99(
            person,
            conf_threshold=self.conf_threshold,
            use_normalized=self.use_normalized,
        )
