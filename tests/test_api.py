"""API health and predict/frame tests (no GPU, no video required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "lstm_available" in body


def test_patients_and_rooms(client) -> None:
    assert client.get("/patients").status_code == 200
    assert client.get("/rooms").status_code == 200


def test_predict_frame(client) -> None:
    xy = [[0.5, 0.2]] * 17
    r = client.post(
        "/predict/frame",
        json={
            "frame_number": 1,
            "timestamp": 0.1,
            "persons": [
                {
                    "person_id": 1,
                    "keypoints_normalized": xy,
                    "keypoint_confidence": [1.0] * 17,
                    "bbox": [10, 10, 80, 200],
                    "detection_confidence": 0.9,
                }
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["person_detected"] is True
    assert body["result"]["skeleton"]["format"] == "coco17"
    risk = client.get("/monitor/state")
    assert risk.status_code == 200
    assert risk.json()["skeleton"]["format"] == "coco17"
