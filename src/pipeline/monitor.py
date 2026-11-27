"""End-to-end frame monitor: pose → features → buffer → LSTM → risk → events."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.alerts.event_manager import EventManager
from src.features.normalize import TrainStatNormalizer
from src.features.coco17_xy import person_to_coco17_xy
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
        tracking_cfg = self.config.get("tracking", {})

        self.patient_id = patient_id
        self.room_id = room_id
        self.zones = self.config.get("zones", {})
        self.max_people = max(1, int(tracking_cfg.get("max_people", 10)))
        self.max_missing_frames = max(
            0, int(tracking_cfg.get("max_missing_frames", 30))
        )
        self.representation = str(model_cfg.get("representation", "upfall33_xyz"))
        self.feature_dim = int(
            model_cfg.get("feature_dim", temporal.get("upfall_feature_dim", 99))
        )
        expected_width = {"upfall33_xyz": 99, "coco17_xy": 34}.get(
            self.representation
        )
        if expected_width is None or self.feature_dim != expected_width:
            raise ValueError(
                f"Unsupported model representation/width: "
                f"{self.representation}/{self.feature_dim}"
            )
        self.selector = PersonSelector(target_person_id)
        self.converter = YoloToUpfallConverter(
            conf_threshold=float(pose_cfg.get("keypoint_conf_threshold", 0.30))
        )
        self.buffers = PersonSequenceStore(
            sequence_length=int(temporal.get("sequence_length", 20)),
            stride=int(temporal.get("stride", 5)),
            feature_dim=self.feature_dim,
        )
        self.risk_engine = RiskEngine(self.config)
        self.events = EventManager()
        self.lstm = LstmPredictor(
            model_path=model_cfg.get("path", "models/upfall_lstm_best.keras"),
            decision_threshold=float(model_cfg.get("decision_threshold", 0.5)),
            fall_class=0,
            feature_dim=self.feature_dim,
        )
        self.normalizer: TrainStatNormalizer | None = None
        self._prev_hip: dict[int, float] = {}
        self._prev_ts: dict[int, float] = {}
        self._prev_level: dict[int, str] = {}
        self._last_lstm_p: dict[int, float | None] = {}
        self._active_people: dict[int, dict[str, Any]] = {}
        self._last_seen_frames: dict[int, int] = {}
        self._last_frame_number: int | None = None
        self.last_risk: RiskResult | None = None
        self.last_signals: PoseSignals | None = None
        self.last_person: PersonPose | None = None
        self.last_status: dict[str, Any] | None = None

        if load_lstm:
            self.lstm.load()
        try:
            self.normalizer = TrainStatNormalizer.from_files(
                mean_path=model_cfg.get("normalization_mean"),
                std_path=model_cfg.get("normalization_std"),
            )
        except FileNotFoundError as exc:
            logger.warning("%s Live LSTM path will skip z-score.", exc)
            self.normalizer = None

    def _forget_person(self, person_id: int) -> None:
        self.buffers.drop(person_id)
        self.risk_engine.forget_person(person_id)
        self.events.forget_person(person_id)
        self._prev_hip.pop(person_id, None)
        self._prev_ts.pop(person_id, None)
        self._prev_level.pop(person_id, None)
        self._last_lstm_p.pop(person_id, None)
        self._active_people.pop(person_id, None)
        self._last_seen_frames.pop(person_id, None)

    def _process_person(
        self,
        person: PersonPose,
        frame_shape: tuple[int, ...] | None,
    ) -> dict[str, Any]:
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

        features = (
            person_to_coco17_xy(person)
            if self.representation == "coco17_xy"
            else self.converter.convert_person(person)
        )
        ready = self.buffers.push(
            person.person_id,
            features,
            person.frame_number,
            person.timestamp,
        )

        lstm_p = self._last_lstm_p.get(person.person_id)
        lstm_ready = ready is not None
        lstm_latency_ms = None
        if ready is not None and self.lstm.available:
            seq = ready.sequence
            if self.normalizer is not None:
                seq = self.normalizer.transform(seq)
            started = time.perf_counter()
            pred = self.lstm.predict_sequence(seq)
            lstm_latency_ms = (time.perf_counter() - started) * 1000.0
            lstm_p = pred["fall_probability"]
            self._last_lstm_p[person.person_id] = lstm_p
            logger.info(
                "LSTM inference person=%s frame=%s fall_p=%s",
                person.person_id,
                person.frame_number,
                lstm_p,
            )
        elif ready is not None and not self.lstm.available:
            logger.debug("Sequence ready but LSTM unavailable: %s", self.lstm.load_error)

        risk = self.risk_engine.score(person.person_id, signals, lstm_p)
        previous_level = self._prev_level.get(person.person_id)
        event = self.events.maybe_emit_from_risk(
            person_id=person.person_id,
            patient_id=self.patient_id,
            room_id=self.room_id,
            zone=zone,
            frame_number=person.frame_number,
            risk_score=risk.risk_score,
            confidence=person.detection_confidence,
            fall_confirmed=risk.fall_confirmed,
            risk_level=risk.risk_level,
            previous_level=previous_level,
        )
        self._prev_level[person.person_id] = risk.risk_level

        if event is not None:
            logger.info(
                "Event %s person=%s score=%.3f",
                event.event_type,
                person.person_id,
                risk.risk_score,
            )

        return {
            "person_id": person.person_id,
            "person_detected": True,
            "tracking_status": "TRACKED",
            "frame_number": person.frame_number,
            "timestamp": person.timestamp,
            "zone": zone,
            "signals": signals.to_dict(),
            "risk": risk.to_dict(),
            "confidence": person.detection_confidence,
            "mean_keypoint_confidence": signals.mean_keypoint_conf,
            "lstm_ready": lstm_ready,
            "lstm_available": self.lstm.available,
            "lstm_latency_ms": (
                None if lstm_latency_ms is None else round(lstm_latency_ms, 2)
            ),
            "events": [] if event is None else [event.to_dict()],
            "patient_id": self.patient_id,
            "room_id": self.room_id,
            "skeleton": {
                "format": "coco17",
                "keypoints_normalized": person.keypoints_normalized.tolist(),
                "keypoint_confidence": person.keypoint_confidence.tolist(),
                "bbox": list(person.bounding_box.as_xyxy),
                "detection_confidence": person.detection_confidence,
            },
        }

    def process_pose_frame(
        self,
        pose: FramePoseResult,
        frame_shape: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if (
            self._last_frame_number is not None
            and pose.frame_number < self._last_frame_number
        ):
            for person_id in list(self._active_people):
                self._forget_person(person_id)
        self._last_frame_number = pose.frame_number

        all_person_ids = [person.person_id for person in pose.persons]
        if len(set(all_person_ids)) != len(all_person_ids):
            raise ValueError("A frame cannot contain duplicate person IDs")

        persons = sorted(
            pose.persons,
            key=lambda item: item.detection_confidence,
            reverse=True,
        )[: self.max_people]
        person_ids = [person.person_id for person in persons]

        current_ids = set(person_ids)
        current_statuses: dict[int, dict[str, Any]] = {}
        for person in persons:
            status = self._process_person(person, frame_shape)
            self._active_people[person.person_id] = status
            self._last_seen_frames[person.person_id] = pose.frame_number
            current_statuses[person.person_id] = status

        people: list[dict[str, Any]] = []
        for person_id, state in list(self._active_people.items()):
            age = pose.frame_number - self._last_seen_frames[person_id]
            if person_id not in current_ids:
                if age > self.max_missing_frames:
                    self._forget_person(person_id)
                    continue
                state = {
                    **state,
                    "person_detected": False,
                    "tracking_status": "LOST",
                    "frames_since_seen": age,
                }
            else:
                state = {
                    **state,
                    "tracking_status": "TRACKED",
                    "frames_since_seen": 0,
                }
            people.append(state)
        people.sort(key=lambda item: int(item["person_id"]))

        selected = self.selector.select(
            FramePoseResult(
                frame_number=pose.frame_number,
                timestamp=pose.timestamp,
                persons=persons,
            )
        )
        primary = next(
            (item for item in people if selected and item["person_id"] == selected.person_id),
            None,
        )
        if primary is None and people:
            primary = next(
                (item for item in people if item["tracking_status"] == "TRACKED"),
                people[0],
            )

        active_events = [
            event
            for item in people
            if item["tracking_status"] == "TRACKED"
            for event in item["events"]
        ]
        lstm_latency = sum(
            item["lstm_latency_ms"] or 0.0
            for item in people
            if item["tracking_status"] == "TRACKED"
        )
        payload: dict[str, Any] = {
            "person_detected": bool(persons),
            "person_count": pose.person_count,
            "tracked_person_count": sum(
                item["tracking_status"] == "TRACKED" for item in people
            ),
            "people": people,
            "frame_number": pose.frame_number,
            "timestamp": pose.timestamp,
            "fps": pose.fps_hint,
            "tracking_mode": pose.tracking_mode,
            "performance": {
                "yolo_inference_ms": (
                    None
                    if pose.inference_latency_ms is None
                    else round(pose.inference_latency_ms, 2)
                ),
                "lstm_inference_ms": round(lstm_latency, 2),
                "pipeline_latency_ms": round(
                    (time.perf_counter() - started) * 1000.0, 2
                ),
                "detected_people": pose.person_count,
                "tracked_people": sum(
                    item["tracking_status"] == "TRACKED" for item in people
                ),
            },
            "risk": None if primary is None else primary["risk"],
            "signals": None if primary is None else primary["signals"],
            "zone": None if primary is None else primary["zone"],
            "person_id": None if primary is None else primary["person_id"],
            "lstm_ready": any(
                item["lstm_ready"] for item in current_statuses.values()
            ),
            "lstm_available": self.lstm.available,
            "events": active_events,
            "patient_id": self.patient_id,
            "room_id": self.room_id,
            "skeleton": None if primary is None else primary["skeleton"],
        }

        if primary is not None:
            self.last_risk = RiskResult(
                **{
                    key: primary["risk"][key]
                    for key in (
                        "risk_score",
                        "risk_level",
                        "fall_confirmed",
                        "lstm_fall_probability",
                        "components",
                        "consecutive_high",
                        "prediction_label",
                    )
                }
            )
            self.last_person = next(
                (person for person in persons if person.person_id == primary["person_id"]),
                self.last_person,
            )
            self.last_signals = (
                None
                if primary["signals"] is None
                else PoseSignals(**primary["signals"])
            )

        self.last_status = payload
        return payload
