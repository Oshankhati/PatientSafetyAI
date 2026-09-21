"""End-to-end frame monitor: pose → features → buffer → LSTM → risk → events."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.alerts.event_manager import EventManager
from src.features.normalize import TrainStatNormalizer
from src.features.pose_signals import PoseSignals, compute_pose_signals
from src.features.sequence_buffer import PersonSequenceStore
from src.features.yolo_to_upfall import YoloToUpfallConverter
from src.pose.schemas import FramePoseResult, PersonPose
from src.risk.risk_engine import RiskEngine, RiskResult
from src.temporal.lstm_predictor import LstmPredictor
from src.tracking.person_selector import PersonSelector
from src.utils.config_loader import load_config
from src.zones.room_zones import locate_zone

logger = logging.getLogger(__name__)


class FrameMonitor:
    """Stateful per-stream monitor. Does not store raw video."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        patient_id: str = "Patient 01",
        room_id: str = "ICU-101",
        target_person_id: int | None = None,
        load_lstm: bool = True,
    ) -> None:
        self.config = config or load_config()
        temporal = self.config.get("temporal", {})
        pose_cfg = self.config.get("pose", {})
        model_cfg = self.config.get("model", {})

        self.patient_id = patient_id
        self.room_id = room_id
        self.zones = self.config.get("zones", {})
        self.selector = PersonSelector(target_person_id)
        self.converter = YoloToUpfallConverter(
            conf_threshold=float(pose_cfg.get("keypoint_conf_threshold", 0.30))
        )
        self.buffers = PersonSequenceStore(
            sequence_length=int(temporal.get("sequence_length", 20)),
            stride=int(temporal.get("stride", 5)),
            feature_dim=int(temporal.get("upfall_feature_dim", 99)),
        )
        self.risk_engine = RiskEngine(self.config)
        self.events = EventManager()
        self.lstm = LstmPredictor(
            model_path=model_cfg.get("path", "models/upfall_lstm_best.keras"),
            decision_threshold=float(model_cfg.get("decision_threshold", 0.5)),
            fall_class=0,
        )
        self.normalizer: TrainStatNormalizer | None = None
        self._prev_hip: dict[int, float] = {}
        self._prev_ts: dict[int, float] = {}
        self._prev_level: dict[int, str] = {}
        self._last_lstm_p: dict[int, float | None] = {}
        self.last_risk: RiskResult | None = None
        self.last_signals: PoseSignals | None = None
        self.last_person: PersonPose | None = None
        self.last_status: dict[str, Any] | None = None

        if load_lstm:
            self.lstm.load()
        try:
            self.normalizer = TrainStatNormalizer.from_files()
        except FileNotFoundError as exc:
            logger.warning("%s Live LSTM path will skip z-score.", exc)
            self.normalizer = None

    def process_pose_frame(
        self,
        pose: FramePoseResult,
        frame_shape: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        person = self.selector.select(pose)
        if person is None:
            empty = {
                "person_detected": False,
                "person_count": pose.person_count,
                "frame_number": pose.frame_number,
                "risk": None,
                "events": [],
                "lstm_ready": False,
                "skeleton": None,
                "patient_id": self.patient_id,
                "room_id": self.room_id,
            }
            self.last_status = empty
            return empty

        wh = None
        if frame_shape is not None and len(frame_shape) >= 2:
            wh = (int(frame_shape[1]), int(frame_shape[0]))
        zone = locate_zone(person, self.zones, wh)

        dt = 1.0
        prev_ts = self._prev_ts.get(person.person_id)
        if prev_ts is not None:
            dt = max(person.timestamp - prev_ts, 1e-3)
        signals = compute_pose_signals(
            person,
            prev_hip_y=self._prev_hip.get(person.person_id),
            dt=dt,
            conf_threshold=self.converter.conf_threshold,
            zone=zone,
        )
        self._prev_hip[person.person_id] = signals.hip_y
        self._prev_ts[person.person_id] = person.timestamp

        features = self.converter.convert_person(person)
        ready = self.buffers.push(
            person.person_id,
            features,
            pose.frame_number,
            pose.timestamp,
        )

        lstm_p = self._last_lstm_p.get(person.person_id)
        lstm_ready = ready is not None
        if ready is not None and self.lstm.available:
            seq = ready.sequence
            if self.normalizer is not None:
                seq = self.normalizer.transform(seq)
            pred = self.lstm.predict_sequence(seq)
            lstm_p = pred["fall_probability"]
            self._last_lstm_p[person.person_id] = lstm_p
            logger.info(
                "LSTM inference person=%s frame=%s fall_p=%s",
                person.person_id,
                pose.frame_number,
                lstm_p,
            )
        elif ready is not None and not self.lstm.available:
            logger.debug("Sequence ready but LSTM unavailable: %s", self.lstm.load_error)

        risk = self.risk_engine.score(person.person_id, signals, lstm_p)
        prev_level = self._prev_level.get(person.person_id)
        event = self.events.maybe_emit_from_risk(
            person_id=person.person_id,
            patient_id=self.patient_id,
            room_id=self.room_id,
            zone=zone,
            frame_number=pose.frame_number,
            risk_score=risk.risk_score,
            confidence=person.detection_confidence,
            fall_confirmed=risk.fall_confirmed,
            risk_level=risk.risk_level,
            previous_level=prev_level,
        )
        self._prev_level[person.person_id] = risk.risk_level
        self.last_risk = risk
        self.last_signals = signals
        self.last_person = person

        if event is not None:
            logger.info("Event %s person=%s score=%.3f", event.event_type, person.person_id, risk.risk_score)

        payload = {
            "person_detected": True,
            "person_count": pose.person_count,
            "person_id": person.person_id,
            "frame_number": pose.frame_number,
            "timestamp": pose.timestamp,
            "zone": zone,
            "signals": signals.to_dict(),
            "risk": risk.to_dict(),
            "lstm_ready": lstm_ready,
            "lstm_available": self.lstm.available,
            "events": [] if event is None else [event.to_dict()],
            "patient_id": self.patient_id,
            "room_id": self.room_id,
            # Normalized skeleton only — not a camera frame (privacy-conscious).
            "skeleton": {
                "format": "coco17",
                "keypoints_normalized": person.keypoints_normalized.tolist(),
                "keypoint_confidence": person.keypoint_confidence.tolist(),
                "bbox": list(person.bounding_box.as_xyxy),
                "detection_confidence": person.detection_confidence,
            },
        }
        self.last_status = payload
        return payload
