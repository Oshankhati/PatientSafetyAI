"""Feature conversion between YOLO COCO-17 and UP-Fall 99-dim formats.

YOLO Pose (COCO-17) and the UP-Fall LSTM (33×XYZ=99) are **not** the same
representation. Conversion is explicit in ``yolo_to_upfall.py`` and remains
an approximation (no depth, joint set mismatch, domain shift).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pose.schemas import UPFALL_FEATURE_DIM
from src.features.yolo_to_upfall import YoloToUpfallConverter, coco17_to_upfall99


@dataclass(frozen=True, slots=True)
class FeatureFormat:
    name: str
    keypoints: int
    dims_per_keypoint: int
    includes_confidence: bool
    coordinate_system: str
    feature_dim: int


YOLO_COCO17 = FeatureFormat(
    name="yolo_coco17",
    keypoints=17,
    dims_per_keypoint=2,
    includes_confidence=True,
    coordinate_system="image_pixel_or_normalized_xy",
    feature_dim=17 * 2,
)

UPFALL_KINECT33 = FeatureFormat(
    name="upfall_kinect33_xyz",
    keypoints=33,
    dims_per_keypoint=3,
    includes_confidence=False,
    coordinate_system="dataset_normalized_xyz",
    feature_dim=UPFALL_FEATURE_DIM,
)


def formats_compatible(a: FeatureFormat, b: FeatureFormat) -> bool:
    return a.name == b.name and a.feature_dim == b.feature_dim


def yolo_person_to_feature_vector(
    keypoints_xy: np.ndarray,
    keypoint_confidence: np.ndarray | None = None,
    include_confidence: bool = True,
) -> np.ndarray:
    """Flatten one YOLO person into an explicit COCO feature vector.

    Default layout:
      [x0,y0,c0, ..., x16,y16,c16]  → length 51
    or without confidence:
      [x0,y0, ..., x16,y16]  → length 34

    This is **not** the UP-Fall 99-dim layout. Use ``YoloToUpfallConverter``.
    """
    kxy = np.asarray(keypoints_xy, dtype=np.float32).reshape(17, 2)
    if not include_confidence:
        return kxy.reshape(-1)

    if keypoint_confidence is None:
        conf = np.ones((17,), dtype=np.float32)
    else:
        conf = np.asarray(keypoint_confidence, dtype=np.float32).reshape(17)

    out = np.zeros((17, 3), dtype=np.float32)
    out[:, :2] = kxy
    out[:, 2] = conf
    return out.reshape(-1)


__all__ = [
    "FeatureFormat",
    "YOLO_COCO17",
    "UPFALL_KINECT33",
    "YoloToUpfallConverter",
    "coco17_to_upfall99",
    "formats_compatible",
    "yolo_person_to_feature_vector",
]
