"""Tests for conversion, sequences, risk, alerts, and API health."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerts.event_manager import EventManager
from src.features.normalize import TrainStatNormalizer
from src.features.fall_dataset import load_fall_dataset_split
from src.features.pose_signals import compute_pose_signals
from src.features.sequence_buffer import RollingSequenceBuffer
from src.features.yolo_to_upfall import YoloToUpfallConverter, coco17_to_upfall99
from src.pipeline.monitor import FrameMonitor
from src.pose.schemas import BoundingBox, FramePoseResult, PersonPose
from src.risk.risk_engine import RiskEngine
from src.temporal.threshold_tune import apply_threshold, select_operating_point, sweep_thresholds
from src.tracking.person_selector import PersonSelector
from src.utils.paths import project_root


def _standing_person(pid: int = 1, frame: int = 0, ts: float = 0.0) -> PersonPose:
    xy = np.zeros((17, 2), dtype=np.float32)
    # normalized standing-ish pose
    xy[0] = [0.5, 0.15]  # nose
    xy[5], xy[6] = [0.4, 0.30], [0.6, 0.30]
    xy[11], xy[12] = [0.42, 0.55], [0.58, 0.55]
    xy[13], xy[14] = [0.42, 0.75], [0.58, 0.75]
    xy[15], xy[16] = [0.42, 0.92], [0.58, 0.92]
    return PersonPose(
        person_id=pid,
        bounding_box=BoundingBox(100, 20, 180, 400, 0.9),
        keypoints=xy * 640,
        keypoint_confidence=np.ones(17, np.float32),
        keypoints_normalized=xy,
        frame_number=frame,
        timestamp=ts,
        detection_confidence=0.9,
    )


def test_coco_to_upfall_shape() -> None:
    xy = np.random.rand(17, 2).astype(np.float32)
    conf = np.ones(17, np.float32)
    vec = coco17_to_upfall99(xy, conf)
    assert vec.shape == (99,)
    # Z channels are 0
    assert np.allclose(vec[2::3], 0.0)


def test_converter_person() -> None:
    conv = YoloToUpfallConverter()
    vec = conv.convert_person(_standing_person())
    assert vec.shape == (99,)
    assert vec[0] != 0 or vec[1] != 0  # pelvis from hips


def test_sequence_buffer_stride() -> None:
    buf = RollingSequenceBuffer(sequence_length=4, stride=2, feature_dim=3)
    outs = []
    for i in range(8):
        ready = buf.push(np.array([i, 0, 0], np.float32), 1, i, float(i))
        if ready is not None:
            outs.append(ready)
    # first emit at frame index 3 (full), then every 2
    assert len(outs) >= 2
    assert outs[0].sequence.shape == (4, 3)


def test_train_normalizer_if_present() -> None:
    split = project_root() / "processed" / "sequences" / "split"
    if not (split / "normalization_mean.npy").exists():
        pytest.skip("normalization stats not on disk")
    n = TrainStatNormalizer.from_files()
    x = np.zeros((20, 99), np.float32)
    y = n.transform(x)
    assert y.shape == (20, 99)


def test_risk_engine_persistence() -> None:
    cfg = {
        "risk": {
            "min_consecutive_frames": 3,
            "fall_threshold": 0.5,
            "high_risk_threshold": 0.5,
            "fall_min": 0.5,
            "low_max": 0.2,
            "medium_max": 0.4,
        },
        "risk_weights": {"lstm": 1.0, "posture": 0, "vertical": 0, "height": 0, "aspect": 0, "zone": 0},
    }
    engine = RiskEngine(cfg)
    person = _standing_person()
    sig = compute_pose_signals(person)
    last = None
    for _ in range(3):
        last = engine.score(1, sig, lstm_fall_probability=0.95)
    assert last is not None
    assert last.fall_confirmed is True
    assert last.consecutive_high >= 3


def test_event_manager_fall_once_then_recovery() -> None:
    ev = EventManager()
    e1 = ev.maybe_emit_from_risk(
        person_id=1,
        patient_id="P",
        room_id="R",
        zone="floor_zone",
        frame_number=10,
        risk_score=0.95,
        confidence=0.8,
        fall_confirmed=True,
        risk_level="FALL DETECTED",
        previous_level="HIGH RISK",
    )
    assert e1 is not None and e1.event_type == "FALL_DETECTED"
    e2 = ev.maybe_emit_from_risk(
        person_id=1,
        patient_id="P",
        room_id="R",
        zone="bed_zone",
        frame_number=40,
        risk_score=0.1,
        confidence=0.8,
        fall_confirmed=False,
        risk_level="LOW RISK",
        previous_level="FALL DETECTED",
    )
    assert e2 is not None and e2.event_type == "RECOVERY"


def test_frame_monitor_without_forcing_lstm() -> None:
    mon = FrameMonitor(load_lstm=False)
    pose = FramePoseResult(0, 0.0, [_standing_person()])
    out = mon.process_pose_frame(pose)
    assert out["person_detected"] is True
    assert out["risk"]["risk_score"] >= 0.0
    assert out["skeleton"]["format"] == "coco17"
    # buffer not full yet
    assert out["lstm_ready"] is False


def test_frame_monitor_keeps_per_person_histories_and_expires_lost_tracks() -> None:
    config = {
        "tracking": {"max_people": 3, "max_missing_frames": 1},
        "temporal": {"sequence_length": 2, "stride": 1, "upfall_feature_dim": 99},
        "model": {"representation": "upfall33_xyz", "feature_dim": 99},
        "zones": {},
    }
    monitor = FrameMonitor(config=config, load_lstm=False)
    first = _standing_person(pid=1, frame=0, ts=0.0)
    second = _standing_person(pid=2, frame=0, ts=0.0)
    second.keypoints_normalized = second.keypoints_normalized + 0.1

    def person(pid: int, frame: int, offset: float = 0.0) -> PersonPose:
        item = _standing_person(pid=pid, frame=frame, ts=frame / 10)
        if offset:
            item.keypoints_normalized = item.keypoints_normalized + offset
        return item

    initial = monitor.process_pose_frame(FramePoseResult(0, 0.0, [first, second]))
    reordered = monitor.process_pose_frame(
        FramePoseResult(
            1,
            0.1,
            [
                person(2, 1, 0.1),
                person(1, 1),
            ],
        )
    )
    assert initial["tracked_person_count"] == 2
    assert [person["person_id"] for person in reordered["people"]] == [1, 2]
    assert set(monitor.buffers._buffers) == {1, 2}
    assert len(monitor.buffers._buffers[1]) == len(monitor.buffers._buffers[2]) == 2
    assert not np.array_equal(
        monitor.buffers._buffers[1]._frames[0],
        monitor.buffers._buffers[2]._frames[0],
    )

    three_people = monitor.process_pose_frame(
        FramePoseResult(
            2,
            0.2,
            [person(1, 2), person(2, 2, 0.1), person(3, 2, 0.2)],
        )
    )
    assert three_people["tracked_person_count"] == 3

    lost = monitor.process_pose_frame(
        FramePoseResult(3, 0.3, [person(1, 3), person(3, 3, 0.2)])
    )
    assert next(item for item in lost["people"] if item["person_id"] == 2)[
        "tracking_status"
    ] == "LOST"
    returned = monitor.process_pose_frame(
        FramePoseResult(
            4,
            0.4,
            [person(1, 4), person(2, 4, 0.1), person(3, 4, 0.2)],
        )
    )
    assert returned["tracked_person_count"] == 3
    assert len(monitor.buffers._buffers[2]) == 2

    monitor.process_pose_frame(FramePoseResult(5, 0.5, [person(1, 5), person(3, 5, 0.2)]))
    expired = monitor.process_pose_frame(
        FramePoseResult(6, 0.6, [person(1, 6), person(3, 6, 0.2)])
    )
    assert [item["person_id"] for item in expired["people"]] == [1, 3]
    assert 2 not in monitor.buffers._buffers
    assert 2 not in monitor.risk_engine._consecutive

    new_entry = monitor.process_pose_frame(
        FramePoseResult(7, 0.7, [person(1, 7), person(3, 7, 0.2), person(4, 7, 0.3)])
    )
    assert [item["person_id"] for item in new_entry["people"]] == [1, 3, 4]
    assert len(monitor.buffers._buffers[4]) == 1

    monitor.process_pose_frame(FramePoseResult(8, 0.8, []))
    gone = monitor.process_pose_frame(FramePoseResult(9, 0.9, []))
    assert gone["people"] == []
    assert gone["tracked_person_count"] == 0
    assert monitor.buffers._buffers == {}


@pytest.mark.parametrize("fall_person_id", [1, 2])
def test_frame_monitor_keeps_predictions_and_risks_isolated(
    fall_person_id: int,
) -> None:
    config = {
        "tracking": {"max_people": 3, "max_missing_frames": 2},
        "temporal": {"sequence_length": 2, "stride": 1, "upfall_feature_dim": 99},
        "model": {"representation": "coco17_xy", "feature_dim": 34},
        "risk": {
            "fall_min": 0.9,
            "fall_threshold": 0.9,
            "high_risk_threshold": 0.7,
            "min_consecutive_frames": 1,
            "recovery_threshold": 0.2,
            "recovery_frames": 2,
        },
        "risk_weights": {
            "lstm": 1.0,
            "posture": 0.0,
            "vertical": 0.0,
            "height": 0.0,
            "aspect": 0.0,
            "zone": 0.0,
        },
        "zones": {},
    }
    monitor = FrameMonitor(config=config, load_lstm=False)
    monitor.normalizer = None

    class PersonSensitivePredictor:
        available = True
        load_error = None

        def __init__(self) -> None:
            self.sequence_means: list[float] = []

        def predict_sequence(self, sequence: np.ndarray) -> dict[str, float]:
            mean = float(sequence.mean())
            self.sequence_means.append(mean)
            return {"fall_probability": 0.95 if mean > 0.5 else 0.05}

    predictor = PersonSensitivePredictor()
    monitor.lstm = predictor

    def person(pid: int, marker: float, frame: int) -> PersonPose:
        item = _standing_person(pid=pid, frame=frame, ts=frame / 10)
        item.keypoints_normalized = np.full((17, 2), marker, dtype=np.float32)
        return item

    for frame in range(2):
        statuses = monitor.process_pose_frame(
            FramePoseResult(
                frame,
                frame / 10,
                [
                    person(1, 0.8 if fall_person_id == 1 else 0.2, frame),
                    person(2, 0.8 if fall_person_id == 2 else 0.2, frame),
                ],
            )
        )

    by_id = {item["person_id"]: item for item in statuses["people"]}
    assert len(predictor.sequence_means) == 2
    assert by_id[fall_person_id]["risk"]["fall_confirmed"] is True
    assert by_id[fall_person_id]["risk"]["lstm_fall_probability"] == 0.95
    normal_id = 2 if fall_person_id == 1 else 1
    assert by_id[normal_id]["risk"]["fall_confirmed"] is False
    assert by_id[normal_id]["risk"]["lstm_fall_probability"] == 0.05
    assert monitor._last_lstm_p == {
        1: 0.95 if fall_person_id == 1 else 0.05,
        2: 0.95 if fall_person_id == 2 else 0.05,
    }


def test_frame_monitor_rejects_duplicate_ids() -> None:
    monitor = FrameMonitor(config={"tracking": {"max_people": 1}}, load_lstm=False)
    pose = FramePoseResult(
        0,
        0.0,
        [_standing_person(pid=4), _standing_person(pid=4)],
    )
    with pytest.raises(ValueError, match="duplicate person IDs"):
        monitor.process_pose_frame(pose)


def test_risk_fall_latch_and_recovery_are_independent_per_person() -> None:
    engine = RiskEngine(
        {
            "risk": {
                "min_consecutive_frames": 2,
                "fall_threshold": 0.9,
                "fall_min": 0.9,
                "high_risk_threshold": 0.7,
                "recovery_threshold": 0.2,
                "recovery_frames": 2,
            },
            "risk_weights": {
                "lstm": 1.0,
                "posture": 0.0,
                "vertical": 0.0,
                "height": 0.0,
                "aspect": 0.0,
                "zone": 0.0,
            },
        }
    )
    signals = compute_pose_signals(_standing_person())
    assert not engine.score(1, signals, 0.99).fall_confirmed
    assert not engine.score(2, signals, 0.99).fall_confirmed
    assert engine.score(1, signals, 0.99).fall_confirmed
    assert engine._fall_counts[2] == 1
    assert not engine.score(2, signals, 0.1).fall_confirmed
    assert engine.score(1, signals, 0.1).fall_confirmed
    assert not engine.score(1, signals, 0.1).fall_confirmed


def test_training_split_excludes_test_labels_when_requested(tmp_path: Path) -> None:
    sequences = tmp_path / "processed" / "sequences"
    sequences.mkdir(parents=True)
    np.save(sequences / "X_sequences.npy", np.zeros((5, 20, 99), np.float32))
    np.save(sequences / "y_sequences.npy", np.array([0, 1, 0, 1, 0], np.int32))
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        {
            "subject": ["SUBJECT1", "SUBJECT2", "SUBJECT3", "SUBJECT4", "SUBJECT5"],
            "file": [f"file{i}" for i in range(5)],
            "start_frame": [0] * 5,
            "end_frame": [20] * 5,
            "label": [0, 1, 0, 1, 999],
        }
    ).to_csv(sequences / "sequence_metadata.csv", index=False)

    split = load_fall_dataset_split("coco17_xy", root=tmp_path, include_test=False)
    assert split.x_train.shape == (3, 20, 34)
    assert split.x_val.shape == (1, 20, 34)
    assert split.x_test.shape == (0, 20, 34)
    assert split.y_test.size == 0
    assert "label" not in split.test_metadata.columns


def test_dataset_split_shapes_if_present() -> None:
    split = project_root() / "processed" / "sequences" / "split"
    xtr = split / "X_train.npy"
    if not xtr.exists():
        pytest.skip("processed split not present")
    x_train = np.load(xtr)
    y_train = np.load(split / "y_train.npy")
    x_test = np.load(split / "X_test.npy")
    assert x_train.shape[1:] == (20, 99)
    assert len(y_train) == x_train.shape[0]
    assert x_test.shape[1:] == (20, 99)
    # subject-wise sizes documented by the project
    assert x_train.shape[0] == 1122
    assert x_test.shape[0] == 272


def test_threshold_sweep_selects_feasible_operating_point() -> None:
    y = np.array([0, 0, 1, 1, 1, 1])
    p1 = np.array([0.2, 0.55, 0.6, 0.8, 0.9, 0.95])
    pred = apply_threshold(p1, 0.5)
    assert pred.tolist() == [0, 1, 1, 1, 1, 1]
    rows = sweep_thresholds(y, p1, thresholds=np.array([0.4, 0.5, 0.7]))
    chosen = select_operating_point(rows, max_false_positive_rate=1.0, min_fall_recall=0.0)
    assert "f1_macro" in chosen
    assert chosen["threshold"] in (0.4, 0.5, 0.7)


def test_threshold_selection_reports_infeasible_constraints() -> None:
    rows = sweep_thresholds(
        np.array([0, 0, 1, 1]),
        np.array([0.9, 0.8, 0.8, 0.9]),
        thresholds=np.array([0.5]),
    )
    selected = select_operating_point(rows)
    assert selected["constraints_met"] is False


def test_person_selector_does_not_mix_ids() -> None:
    sel = PersonSelector()
    a = _standing_person(pid=2, frame=0)
    b = _standing_person(pid=9, frame=1)
    first = sel.select(FramePoseResult(0, 0.0, [a]))
    assert first is not None and first.person_id == 2
    missing = sel.select(FramePoseResult(1, 0.1, [b]))
    assert missing is None


def test_pre_fall_heuristic_event() -> None:
    ev = EventManager()
    e = ev.maybe_emit_from_risk(
        person_id=1,
        patient_id="P",
        room_id="R",
        zone="safe_zone",
        frame_number=3,
        risk_score=0.5,
        confidence=0.8,
        fall_confirmed=False,
        risk_level="MEDIUM RISK",
        previous_level="LOW RISK",
    )
    assert e is not None and e.event_type == "PRE_FALL"


@pytest.mark.slow
def test_lstm_predicts_if_model_present() -> None:
    from src.temporal.lstm_predictor import LstmPredictor

    pred = LstmPredictor()
    pred.load()
    if not pred.available:
        pytest.skip(pred.load_error or "no lstm")
    split = project_root() / "processed" / "sequences" / "split"
    x = np.load(split / "X_test_normalized.npy")[:1]
    out = pred.predict_sequence(x[0])
    assert out["fall_probability"] is not None
    assert 0.0 <= out["fall_probability"] <= 1.0
