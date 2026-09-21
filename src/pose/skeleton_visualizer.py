"""Skeleton and bounding-box visualization helpers."""

from __future__ import annotations

import cv2
import numpy as np

from src.pose.schemas import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    FramePoseResult,
    PersonPose,
)


class SkeletonVisualizer:
    """Draw COCO-17 skeletons and person boxes onto BGR frames."""

    def __init__(
        self,
        keypoint_conf_threshold: float = 0.30,
        box_color: tuple[int, int, int] = (40, 180, 40),
        joint_color: tuple[int, int, int] = (0, 200, 255),
        bone_color: tuple[int, int, int] = (255, 160, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.box_color = box_color
        self.joint_color = joint_color
        self.bone_color = bone_color
        self.text_color = text_color

    def draw_person(self, frame: np.ndarray, person: PersonPose) -> np.ndarray:
        """Draw one person's box + skeleton in-place and return the frame."""
        x1, y1, x2, y2 = map(int, person.bounding_box.as_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)

        label = (
            f"ID {person.person_id} "
            f"{person.detection_confidence:.2f}"
        )
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.text_color,
            2,
            cv2.LINE_AA,
        )

        kpts = person.keypoints
        conf = person.keypoint_confidence
        if kpts is None or len(kpts) == 0:
            return frame

        for a, b in COCO_SKELETON:
            if a >= len(kpts) or b >= len(kpts):
                continue
            if conf[a] < self.keypoint_conf_threshold:
                continue
            if conf[b] < self.keypoint_conf_threshold:
                continue
            pt1 = (int(kpts[a][0]), int(kpts[a][1]))
            pt2 = (int(kpts[b][0]), int(kpts[b][1]))
            if pt1 == (0, 0) or pt2 == (0, 0):
                continue
            cv2.line(frame, pt1, pt2, self.bone_color, 2, cv2.LINE_AA)

        for i, (x, y) in enumerate(kpts):
            if conf[i] < self.keypoint_conf_threshold:
                continue
            if int(x) == 0 and int(y) == 0:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, self.joint_color, -1, cv2.LINE_AA)

        return frame

    def draw_frame(
        self,
        frame: np.ndarray,
        pose_result: FramePoseResult,
        overlay_text: list[str] | None = None,
    ) -> np.ndarray:
        """Draw all persons and optional HUD text. Does not copy unless needed."""
        out = frame
        for person in pose_result.persons:
            self.draw_person(out, person)

        lines = overlay_text or []
        y = 28
        for line in lines:
            cv2.putText(
                out,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                self.text_color,
                1,
                cv2.LINE_AA,
            )
            y += 26
        return out

    @staticmethod
    def keypoint_legend() -> str:
        return ", ".join(
            f"{i}:{name}" for i, name in enumerate(COCO_KEYPOINT_NAMES)
        )
