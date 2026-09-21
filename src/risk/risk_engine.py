"""Prototype risk estimation. Not medically validated.

risk_score =
    w_lstm      * lstm_fall_probability
  + w_posture   * clip(torso_angle_deg / 90)
  + w_vertical  * clip(downward_hip_velocity / v_ref)
  + w_height    * (1 - clip(body_height / h_ref))
  + w_aspect    * clip((bbox_aspect - 0.6) / 1.2)
  + w_zone      * zone_prior

Missing LSTM: remaining weights are renormalized to sum to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.features.pose_signals import PoseSignals

LEVEL_LOW = "LOW RISK"
LEVEL_MEDIUM = "MEDIUM RISK"
LEVEL_HIGH = "HIGH RISK"
LEVEL_FALL = "FALL DETECTED"


@dataclass
class RiskResult:
    risk_score: float
    risk_level: str
    fall_confirmed: bool
    lstm_fall_probability: float | None
    components: dict[str, float] = field(default_factory=dict)
    consecutive_high: int = 0
    prediction_label: str = "NORMAL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "fall_confirmed": self.fall_confirmed,
            "lstm_fall_probability": self.lstm_fall_probability,
            "components": self.components,
            "consecutive_high": self.consecutive_high,
            "prediction_label": self.prediction_label,
        }


class RiskEngine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        risk = cfg.get("risk", {})
        weights = cfg.get("risk_weights", {})
        self.low_max = float(risk.get("low_max", 0.39))
        self.medium_max = float(risk.get("medium_max", 0.69))
        self.high_max = float(risk.get("high_max", 0.89))
        self.fall_min = float(risk.get("fall_min", 0.90))
        self.min_consecutive = int(risk.get("min_consecutive_frames", 5))
        self.high_risk_threshold = float(risk.get("high_risk_threshold", 0.70))
        self.fall_threshold = float(risk.get("fall_threshold", 0.90))

        self.w_lstm = float(weights.get("lstm", 0.35))
        self.w_posture = float(weights.get("posture", 0.20))
        self.w_vertical = float(weights.get("vertical", 0.15))
        self.w_height = float(weights.get("height", 0.15))
        self.w_aspect = float(weights.get("aspect", 0.10))
        self.w_zone = float(weights.get("zone", 0.05))

        self.v_ref = float(weights.get("vertical_ref", 0.35))
        self.h_ref = float(weights.get("height_ref", 0.35))

        self.zone_prior = {
            "floor_zone": 0.85,
            "exit_zone": 0.45,
            "bed_zone": 0.15,
            "safe_zone": 0.10,
            "unknown": 0.20,
        }
        self._consecutive: dict[int, int] = {}

    def _clip01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def score(
        self,
        person_id: int,
        signals: PoseSignals,
        lstm_fall_probability: float | None,
    ) -> RiskResult:
        posture = self._clip01(signals.torso_angle_deg / 90.0)
        vertical = self._clip01(max(signals.vertical_velocity, 0.0) / self.v_ref)
        height_drop = self._clip01(1.0 - (signals.body_height / max(self.h_ref, 1e-6)))
        aspect = self._clip01((signals.bbox_aspect - 0.6) / 1.2)
        zone = self.zone_prior.get(signals.zone, 0.20)

        parts: dict[str, float] = {
            "posture": posture,
            "vertical": vertical,
            "height_drop": height_drop,
            "aspect": aspect,
            "zone": zone,
        }
        weights = {
            "posture": self.w_posture,
            "vertical": self.w_vertical,
            "height_drop": self.w_height,
            "aspect": self.w_aspect,
            "zone": self.w_zone,
        }
        if lstm_fall_probability is not None:
            parts["lstm"] = self._clip01(lstm_fall_probability)
            weights["lstm"] = self.w_lstm

        total_w = sum(weights.values()) or 1.0
        risk = sum(parts[k] * (weights[k] / total_w) for k in parts)

        if risk >= self.fall_min:
            level = LEVEL_FALL
        elif risk >= self.high_risk_threshold:
            level = LEVEL_HIGH
        elif risk > self.low_max and risk <= self.medium_max:
            level = LEVEL_MEDIUM
        elif risk > self.medium_max:
            level = LEVEL_HIGH
        else:
            level = LEVEL_LOW

        if risk >= self.high_risk_threshold:
            self._consecutive[person_id] = self._consecutive.get(person_id, 0) + 1
        else:
            self._consecutive[person_id] = 0

        consec = self._consecutive.get(person_id, 0)
        fall_confirmed = risk >= self.fall_threshold and consec >= self.min_consecutive

        if fall_confirmed:
            label = "FALL DETECTED"
            level = LEVEL_FALL
        elif level in (LEVEL_HIGH, LEVEL_FALL):
            label = "HIGH RISK"
        elif level == LEVEL_MEDIUM:
            label = "PRE-FALL / MEDIUM"  # heuristic state, not a trained class
        else:
            label = "NORMAL"

        return RiskResult(
            risk_score=round(risk, 4),
            risk_level=level,
            fall_confirmed=fall_confirmed,
            lstm_fall_probability=(
                None if lstm_fall_probability is None else round(float(lstm_fall_probability), 4)
            ),
            components={k: round(v, 4) for k, v in parts.items()},
            consecutive_high=consec,
            prediction_label=label,
        )
