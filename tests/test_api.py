"""API health and predict/frame tests (no GPU, no video required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from starlette.websockets import WebSocketDisconnect

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PATIENTSAFETYAI_SECRET_KEY", "local-test-secret-key-not-for-production")
    from backend import database

    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "test.db")
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
    token = _signup_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/patients", headers=headers).status_code == 200
    assert client.get("/rooms", headers=headers).status_code == 200


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
                },
                {
                    "person_id": 2,
                    "keypoints_normalized": xy,
                    "keypoint_confidence": [1.0] * 17,
                    "bbox": [100, 10, 180, 220],
                    "detection_confidence": 0.8,
                },
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["person_detected"] is True
    assert body["result"]["skeleton"]["format"] == "coco17"
    assert body["result"]["tracked_person_count"] == 2
    assert {person["person_id"] for person in body["result"]["people"]} == {1, 2}

    duplicate = client.post(
        "/predict/frame",
        json={
            "frame_number": 2,
            "timestamp": 0.2,
            "persons": [
                {"person_id": 1, "keypoints_normalized": xy},
                {"person_id": 1, "keypoints_normalized": xy},
            ],
        },
    )
    assert duplicate.status_code == 422
    token = _signup_and_login(client)
    risk = client.get("/monitor/state", headers={"Authorization": f"Bearer {token}"})
    assert risk.status_code == 200
    assert risk.json()["skeleton"]["format"] == "coco17"
    with client.websocket_connect(f"/ws/monitor?token={token}") as socket:
        snapshot = socket.receive_json()
        assert snapshot["type"] == "snapshot"
        assert {person["person_id"] for person in snapshot["result"]["people"]} == {1, 2}

    single_person = client.post(
        "/predict/frame",
        json={
            "frame_number": 2,
            "timestamp": 0.2,
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
    assert single_person.status_code == 200
    single_result = single_person.json()["result"]
    assert single_result["tracked_person_count"] == 1
    active_ids = {
        person["person_id"]
        for person in single_result["people"]
        if person["tracking_status"] != "LOST"
    }
    assert active_ids == {1}

    empty_frame = client.post(
        "/predict/frame",
        json={"frame_number": 3, "timestamp": 0.3, "persons": []},
    )
    assert empty_frame.status_code == 200
    result = empty_frame.json()["result"]
    assert result["person_detected"] is False
    assert result["tracked_person_count"] == 0
    assert {person["tracking_status"] for person in result["people"]} == {"LOST"}


def test_staff_signup_and_login(client) -> None:
    token = _signup_and_login(client)
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["staff"]["staff_id"] == "TEST-100"


def _signup_and_login(client) -> str:
    signup = client.post(
        "/auth/signup",
        json={
            "full_name": "Test Nurse",
            "staff_id": "TEST-100",
            "email": "test.nurse@example.com",
            "department": "ICU",
            "password": "test-password-123",
        },
    )
    assert signup.status_code == 200

    login = client.post(
        "/auth/login",
        json={"staff_id": "TEST-100", "password": "test-password-123"},
    )
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert login.json()["staff"]["staff_id"] == "TEST-100"
    return login.json()["access_token"]


def test_dashboard_endpoints_require_auth_and_persist(client) -> None:
    assert client.get("/patients").status_code == 401
    token = _signup_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}

    settings = client.get("/settings", headers=headers)
    assert settings.status_code == 200
    payload = settings.json()["settings"]
    payload["ai"]["medium"] = "0.45"
    saved = client.put("/settings", headers=headers, json={"settings": payload})
    assert saved.status_code == 200
    assert client.get("/settings", headers=headers).json()["settings"]["ai"]["medium"] == "0.45"

    ack = client.post("/events/event-1/acknowledge", headers=headers)
    assert ack.status_code == 200
    assert ack.json()["acknowledgement"]["staff_id"] == "TEST-100"
    stored = client.get("/events/acknowledgements", headers=headers)
    assert stored.json()["acknowledgements"]["event-1"]["staff_name"] == "Test Nurse"


def test_monitor_websocket_requires_valid_session(client) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/monitor"):
            pass

    token = _signup_and_login(client)
    with client.websocket_connect(f"/ws/monitor?token={token}") as socket:
        assert socket.receive_json()["type"] == "snapshot"
