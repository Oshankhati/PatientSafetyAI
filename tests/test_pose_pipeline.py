"""Unit tests for pose schemas, feature bridge, and detector loading."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.format_bridge import (
    UPFALL_KINECT33,
    YOLO_COCO17,
    YoloToUpfallConverter,
    formats_compatible,
    yolo_person_to_feature_vector,
)
from src.pose.keypoint_extractor import KeypointExtractor
from src.pose.schemas import BoundingBox, FramePoseResult, PersonPose
from src.pose.skeleton_visualizer import SkeletonVisualizer
from src.tracking.iou_tracker import IoUTracker
from src.utils.config_loader import load_config
from src.utils.paths import project_root


def test_project_root_exists() -> None:
    root = project_root()
    assert (root / "config" / "config.yaml").exists()
    assert (root / "src" / "pose").is_dir()


def test_load_config() -> None:
    cfg = load_config()
    assert "pose" in cfg
    assert cfg["temporal"]["sequence_length"] == 20
    assert cfg["temporal"]["upfall_feature_dim"] == 99


def test_feature_formats_incompatible() -> None:
    assert not formats_compatible(YOLO_COCO17, UPFALL_KINECT33)
    assert YOLO_COCO17.keypoints == 17
    assert UPFALL_KINECT33.feature_dim == 99


def test_yolo_feature_vector_shapes() -> None:
    kxy = np.random.rand(17, 2).astype(np.float32)
    conf = np.ones(17, dtype=np.float32)
    with_conf = yolo_person_to_feature_vector(kxy, conf, include_confidence=True)
    without = yolo_person_to_feature_vector(kxy, conf, include_confidence=False)
    assert with_conf.shape == (51,)
    assert without.shape == (34,)


def test_upfall_converter_explicit_99_dim() -> None:
    converter = YoloToUpfallConverter()
    vec = converter.convert(np.ones((17, 2), dtype=np.float32), np.ones(17, dtype=np.float32))
    assert vec.shape == (99,)
    assert vec.dtype == np.float32


def test_frame_person_selection() -> None:
    p1 = PersonPose(
        person_id=7,
        bounding_box=BoundingBox(0, 0, 10, 10, 0.5),
        keypoints=np.zeros((17, 2), np.float32),
        keypoint_confidence=np.ones(17, np.float32),
        keypoints_normalized=np.zeros((17, 2), np.float32),
        frame_number=1,
        timestamp=0.1,
        detection_confidence=0.5,
    )
    p2 = PersonPose(
        person_id=3,
        bounding_box=BoundingBox(0, 0, 100, 100, 0.9),
        keypoints=np.zeros((17, 2), np.float32),
        keypoint_confidence=np.ones(17, np.float32),
        keypoints_normalized=np.zeros((17, 2), np.float32),
        frame_number=1,
        timestamp=0.1,
        detection_confidence=0.9,
    )
    result = FramePoseResult(frame_number=1, timestamp=0.1, persons=[p1, p2])
    assert result.select_person(person_id=7).person_id == 7
    assert result.select_person().person_id == 3  # largest box


def test_iou_tracker_keeps_ids_when_detection_order_changes() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_missing_frames=1, id_start=100)
    left = BoundingBox(0, 0, 20, 20)
    right = BoundingBox(50, 50, 70, 70)
    initial = tracker.update([left, right], frame_number=0)
    reordered = tracker.update([right, left], frame_number=1)
    assert initial == [100, 101]
    assert reordered == [101, 100]

    tracker.update([], frame_number=3)
    fresh = tracker.update([left], frame_number=4)
    assert fresh == [102]


def test_iou_tracker_uses_motion_through_a_crossing() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_missing_frames=3, id_start=200)
    centers = ((10, 70), (17, 63), (24, 56), (31, 49), (39, 41), (47, 33))
    for frame_number, positions in enumerate(centers):
        boxes = [
            BoundingBox(center - 10, 10, center + 10, 50)
            for center in positions
        ]
        assert tracker.update(boxes, frame_number) == [200, 201]


def test_visualizer_draws_without_error() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    person = PersonPose(
        person_id=1,
        bounding_box=BoundingBox(20, 20, 100, 200, 0.8),
        keypoints=np.array([[40, 40]] * 17, dtype=np.float32),
        keypoint_confidence=np.ones(17, dtype=np.float32),
        keypoints_normalized=np.zeros((17, 2), dtype=np.float32),
        frame_number=0,
        timestamp=0.0,
        detection_confidence=0.8,
    )
    result = FramePoseResult(0, 0.0, [person])
    out = SkeletonVisualizer().draw_frame(frame.copy(), result, ["test"])
    assert out.shape == frame.shape


def test_visualizer_matches_status_to_each_person_id(monkeypatch) -> None:
    people = [
        PersonPose(
            person_id=1,
            bounding_box=BoundingBox(0, 0, 20, 40),
            keypoints=np.zeros((17, 2), dtype=np.float32),
            keypoint_confidence=np.ones(17, dtype=np.float32),
            keypoints_normalized=np.zeros((17, 2), dtype=np.float32),
            frame_number=0,
            timestamp=0.0,
        ),
        PersonPose(
            person_id=2,
            bounding_box=BoundingBox(30, 0, 50, 40),
            keypoints=np.zeros((17, 2), dtype=np.float32),
            keypoint_confidence=np.ones(17, dtype=np.float32),
            keypoints_normalized=np.zeros((17, 2), dtype=np.float32),
            frame_number=0,
            timestamp=0.0,
        ),
    ]
    result = FramePoseResult(0, 0.0, people)
    statuses = [
        {"person_id": 2, "risk": {"prediction_label": "FALL DETECTED"}},
        {"person_id": 1, "risk": {"prediction_label": "NORMAL"}},
    ]
    visualizer = SkeletonVisualizer()
    received = {}

    def capture_status(frame, person, status):
        received[person.person_id] = status
        return frame

    monkeypatch.setattr(visualizer, "draw_person", capture_status)
    visualizer.draw_frame(np.zeros((60, 60, 3), dtype=np.uint8), result, person_status=statuses)

    assert received[1] == statuses[1]
    assert received[2] == statuses[0]


def test_extractor_empty_result() -> None:
    class _Empty:
        boxes = None
        keypoints = None

    out = KeypointExtractor().extract(_Empty(), frame_number=0, timestamp=0.0)
    assert out.person_count == 0


def test_pose_model_file_present() -> None:
    weights = project_root() / "models" / "yolo11n-pose.pt"
    assert weights.exists(), "Expected models/yolo11n-pose.pt for CPU pose demos"


@pytest.mark.slow
def test_pose_detector_cpu_on_blank_frame() -> None:
    from src.pose.pose_detector import PoseDetector

    detector = PoseDetector.from_config()
    detector.load_model()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.process_frame(frame)
    assert isinstance(result.persons, list)
