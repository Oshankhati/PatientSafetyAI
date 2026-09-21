"""In-memory alert / event log. Does not store raw video."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SafetyEvent:
    event_type: str
    timestamp: str
    patient_id: str
    person_id: int
    confidence: float
    risk_score: float
    room_id: str
    zone: str
    frame_number: int
    snapshot_saved: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventManager:
    def __init__(self, max_events: int = 500) -> None:
        self.max_events = max_events
        self._events: list[SafetyEvent] = []
        self._last_type: dict[int, str] = {}

    def emit(self, event: SafetyEvent) -> SafetyEvent:
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]
        self._last_type[event.person_id] = event.event_type
        return event

    def maybe_emit_from_risk(
        self,
        *,
        person_id: int,
        patient_id: str,
        room_id: str,
        zone: str,
        frame_number: int,
        risk_score: float,
        confidence: float,
        fall_confirmed: bool,
        risk_level: str,
        previous_level: str | None,
    ) -> SafetyEvent | None:
        event_type: str | None = None
        if fall_confirmed:
            event_type = "FALL_DETECTED"
        elif risk_level == "HIGH RISK" and previous_level != "HIGH RISK":
            event_type = "HIGH_RISK"
        elif risk_level == "MEDIUM RISK" and previous_level in (None, "LOW RISK"):
            event_type = "PRE_FALL"  # heuristic state, not a trained class
        elif previous_level is None:
            event_type = "PATIENT_MOVEMENT"
        elif (
            previous_level in ("HIGH RISK", "FALL DETECTED", "MEDIUM RISK")
            and risk_level == "LOW RISK"
        ):
            event_type = "RECOVERY"
        elif zone in ("exit_zone",) and previous_level not in (None,):
            event_type = "PATIENT_LEFT_SAFE_ZONE"

        if event_type is None:
            return None
        if self._last_type.get(person_id) == event_type and event_type != "FALL_DETECTED":
            return None

        event = SafetyEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            patient_id=patient_id,
            person_id=person_id,
            confidence=confidence,
            risk_score=risk_score,
            room_id=room_id,
            zone=zone,
            frame_number=frame_number,
        )
        return self.emit(event)

    def latest(self, n: int = 20) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events[-n:]]

    def all(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events]
