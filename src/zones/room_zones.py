"""Room zone helpers. Coordinates are normalized [x1, y1, x2, y2] in image space."""

from __future__ import annotations

from typing import Any

from src.pose.schemas import PersonPose


def _in_rect(x: float, y: float, rect: list[float]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def hip_center_normalized(person: PersonPose) -> tuple[float, float]:
    """Approximate hip center from bbox if hips missing."""
    k = person.keypoints_normalized
    conf = person.keypoint_confidence
    hips = []
    for idx in (11, 12):
        if idx < len(k) and conf[idx] >= 0.2:
            hips.append(k[idx])
    if hips:
        import numpy as np

        m = np.mean(hips, axis=0)
        return float(m[0]), float(m[1])
    box = person.bounding_box
    # Fallback: lower-middle of box, converted roughly if box is pixels.
    # Callers should pass a person whose keypoints_normalized are in [0,1].
    cx = (box.x1 + box.x2) / 2.0
    cy = box.y1 + 0.7 * (box.y2 - box.y1)
    return float(cx), float(cy)


def locate_zone(person: PersonPose, zones: dict[str, Any], frame_wh: tuple[int, int] | None = None) -> str:
    """Return the most specific zone containing the hip center.

    Priority: floor_zone, bed_zone, exit_zone, safe_zone, unknown.
    Boxes in config are normalized; if hip coords look like pixels, normalize.
    """
    x, y = hip_center_normalized(person)
    if frame_wh and (x > 1.5 or y > 1.5):
        w, h = frame_wh
        x, y = x / max(w, 1), y / max(h, 1)

    # Floor first: lying near the bottom of the frame is a safety concern.
    order = ("floor_zone", "bed_zone", "exit_zone", "safe_zone")
    for name in order:
        rect = zones.get(name)
        if rect and _in_rect(x, y, list(rect)):
            return name
    return "unknown"
