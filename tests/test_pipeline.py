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


def test_threshold_sweep_prefers_macro_f1() -> None:
    y = np.array([0, 0, 1, 1, 1, 1])
    p1 = np.array([0.2, 0.55, 0.6, 0.8, 0.9, 0.95])
    pred = apply_threshold(p1, 0.5)
    assert pred.tolist() == [0, 1, 1, 1, 1, 1]
    rows = sweep_thresholds(y, p1, thresholds=np.array([0.4, 0.5, 0.7]))
    chosen = select_operating_point(rows, max_false_positive_rate=1.0, min_fall_recall=0.0)
    assert "f1_macro" in chosen
    assert chosen["threshold"] in (0.4, 0.5, 0.7)


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
