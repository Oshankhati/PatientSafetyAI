"""Pydantic API models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    project: str
    lstm_available: bool
    yolo_model: str
    note: str = "Academic prototype - not a medical device."


class KeypointPerson(BaseModel):
    person_id: int = 0
    keypoints_normalized: list[list[float]]
    keypoint_confidence: list[float] | None = None
    bbox: list[float] | None = Field(
        default=None, description="Optional [x1,y1,x2,y2] in pixels"
    )
    detection_confidence: float = 0.8


class FramePredictRequest(BaseModel):
    frame_number: int = 0
    timestamp: float = 0.0
    persons: list[KeypointPerson]
    patient_id: str | None = None
    room_id: str | None = None


class VideoPredictRequest(BaseModel):
    video_path: str
    max_frames: int | None = 60
    frame_skip: int = 0
    patient_id: str = "Patient 01"
    room_id: str = "ICU-101"


class RoomConfig(BaseModel):
    id: str
    name: str | None = None
    patient_id: str | None = None
    zones: dict[str, list[float]] | None = None


class PredictResponse(BaseModel):
    ok: bool
    result: dict[str, Any]
