"""FastAPI inference / event service for PatientSafetyAI."""

from __future__ import annotations
from starlette.concurrency import run_in_threadpool
import asyncio
import logging
import threading
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.auth import router as auth_router
from backend.database import initialize_database
from backend.schemas import (
    FramePredictRequest,
    HealthResponse,
    PredictResponse,
    RoomConfig,
    VideoPredictRequest,
)
from src.pipeline.monitor import FrameMonitor
from src.pose.pose_detector import PoseDetector
from src.pose.schemas import BoundingBox, FramePoseResult, PersonPose
from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging
from src.utils.paths import resolve_path

setup_logging()
logger = logging.getLogger("backend")

_monitor: FrameMonitor | None = None
_detector: PoseDetector | None = None
_rooms: list[dict[str, Any]] = []
_patients: list[dict[str, Any]] = []
_ws_clients: list[WebSocket] = []
_camera_thread: threading.Thread | None = None
_camera_stop_event: threading.Event | None = None
_camera_status: dict[str, Any] = {
    "running": False,
    "frames_processed": 0,
    "last_error": None,
}

def get_monitor() -> FrameMonitor:
    if _monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    return _monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _monitor, _rooms, _patients
    initialize_database()
    cfg = load_config()
    _monitor = FrameMonitor(cfg)
#     # _rooms = list(cfg.get("rooms") or [{"id": "ICU-101", "name": "ICU Bay 101"}])
#     default_rooms = [{"id": "ICU-101", "name": "ICU Bay 101"}]
# global_zones = cfg.get("zones") or {}

# _rooms = [
#     {
#         **room,
#         "zones": room.get("zones") or global_zones
#     }
#     for room in (cfg.get("rooms") or default_rooms)
# ]


    default_rooms = [{"id": "ICU-101", "name": "ICU Bay 101"}]
    global_zones = cfg.get("zones") or {}

    _rooms = [
        {
            **room,
            "zones": room.get("zones") or global_zones
        }
        for room in (cfg.get("rooms") or default_rooms)
    ]

    _patients = list(
        cfg.get("patients")
        or [{"id": "Patient 01", "room_id": "ICU-101", "status": "MONITORING"}]
    )
    logger.info("API ready | LSTM available=%s", _monitor.lstm.available)
    yield


app = FastAPI(
    title="PatientSafetyAI",
    description="Privacy-conscious fall-risk monitoring prototype (academic).",
    version="0.3.0",
    lifespan=lifespan,
)
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg = load_config()
    mon = get_monitor()
    return HealthResponse(
        status="ok",
        project=cfg.get("project", {}).get("name", "PatientSafetyAI"),
        lstm_available=mon.lstm.available,
        yolo_model=str(cfg.get("pose", {}).get("model_path", "")),
    )


@app.get("/risk")
@app.get("/monitor/state")
def get_risk() -> dict[str, Any]:
    mon = get_monitor()
    if mon.last_status is not None:
        return mon.last_status
    if mon.last_risk is None:
        return {"risk": None, "message": "No frames processed yet.", "skeleton": None}
    return {
        "patient_id": mon.patient_id,
        "room_id": mon.room_id,
        "risk": mon.last_risk.to_dict(),
        "signals": None if mon.last_signals is None else mon.last_signals.to_dict(),
        "skeleton": None
        if mon.last_person is None
        else {
            "format": "coco17",
            "keypoints_normalized": mon.last_person.keypoints_normalized.tolist(),
            "keypoint_confidence": mon.last_person.keypoint_confidence.tolist(),
        },
    }


@app.get("/events")
def get_events() -> dict[str, Any]:
    return {"events": get_monitor().events.all()}


@app.get("/events/latest")
def get_events_latest(n: int = 20) -> dict[str, Any]:
    return {"events": get_monitor().events.latest(n)}


@app.get("/patients")
def get_patients() -> dict[str, Any]:
    return {"patients": _patients}


@app.get("/rooms")
def get_rooms() -> dict[str, Any]:
    return {"rooms": _rooms}


@app.post("/rooms")
def post_room(room: RoomConfig) -> dict[str, Any]:
    entry = room.model_dump()
    _rooms.append(entry)
    return {"ok": True, "room": entry}


def _person_from_request(item, frame_number: int, timestamp: float) -> PersonPose:
    import numpy as np

    kxy = np.asarray(item.keypoints_normalized, dtype=np.float32)
    if kxy.shape != (17, 2):
        raise HTTPException(status_code=400, detail="keypoints_normalized must be 17x2")
    conf = (
        np.ones(17, dtype=np.float32)
        if item.keypoint_confidence is None
        else np.asarray(item.keypoint_confidence, dtype=np.float32)
    )
    if conf.shape != (17,):
        raise HTTPException(status_code=400, detail="keypoint_confidence must have length 17")
    if item.bbox and len(item.bbox) == 4:
        x1, y1, x2, y2 = item.bbox
    else:
        x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
    return PersonPose(
        person_id=item.person_id,
        bounding_box=BoundingBox(x1, y1, x2, y2, item.detection_confidence),
        keypoints=kxy,
        keypoint_confidence=conf,
        keypoints_normalized=kxy,
        frame_number=frame_number,
        timestamp=timestamp,
        detection_confidence=item.detection_confidence,
    )


# @app.post("/predict", response_model=PredictResponse)
# @app.post("/predict/frame", response_model=PredictResponse)
# def predict_frame(body: FramePredictRequest) -> PredictResponse:
#     if not body.persons:
#         raise HTTPException(status_code=400, detail="No persons provided")
#     persons = [
#         _person_from_request(p, body.frame_number, body.timestamp) for p in body.persons
#     ]
#     pose = FramePoseResult(
#         frame_number=body.frame_number,
#         timestamp=body.timestamp,
#         persons=persons,
#     )
#     mon = get_monitor()
#     if body.patient_id:
#         mon.patient_id = body.patient_id
#     if body.room_id:
#         mon.room_id = body.room_id
#     result = mon.process_pose_frame(pose)
#     _broadcast({"type": "update", "result": result})
#     return PredictResponse(ok=True, result=result)

@app.post("/predict", response_model=PredictResponse)
@app.post("/predict/frame", response_model=PredictResponse)
async def predict_frame(body: FramePredictRequest) -> PredictResponse:
    if not body.persons:
        raise HTTPException(status_code=400, detail="No persons provided")

    persons = [
        _person_from_request(p, body.frame_number, body.timestamp)
        for p in body.persons
    ]

    pose = FramePoseResult(
        frame_number=body.frame_number,
        timestamp=body.timestamp,
        persons=persons,
    )

    mon = get_monitor()

    if body.patient_id:
        mon.patient_id = body.patient_id
    if body.room_id:
        mon.room_id = body.room_id

    result = await run_in_threadpool(mon.process_pose_frame, pose)

    await _broadcast({"type": "update", "result": result})

    return PredictResponse(ok=True, result=result)
# def _broadcast(payload: dict[str, Any]) -> None:
#     if not _ws_clients:
#         return
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         return
#     for ws in list(_ws_clients):
#         loop.create_task(_safe_send(ws, payload))

async def _broadcast(payload: dict[str, Any]) -> None:
    for ws in list(_ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            if ws in _ws_clients:
                _ws_clients.remove(ws)

async def _safe_send(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        if ws in _ws_clients:
            _ws_clients.remove(ws)

def _camera_worker(
    source: int,
    stop_event: threading.Event,
    loop: asyncio.AbstractEventLoop,
) -> None:
    global _camera_status

    try:
        detector = _detector
        monitor = get_monitor()

        if detector is None:
            raise RuntimeError("YOLO detector is not initialized")

        for frame, pose in detector.process_webcam(
            source=source,
            display=False,
        ):
            if stop_event.is_set():
                break

            result = monitor.process_pose_frame(
                pose,
                frame_shape=frame.shape,
            )

            _camera_status["frames_processed"] = pose.frame_number + 1

            try:
                asyncio.run_coroutine_threadsafe(
                    _broadcast({"type": "update", "result": result}),
                    loop,
                )
            except RuntimeError:
                logger.exception("Could not send camera update")

    except Exception as exc:
        logger.exception("Camera worker failed")
        _camera_status["last_error"] = str(exc)

    finally:
        _camera_status["running"] = False


def _initialize_detector() -> None:
    global _detector

    if _detector is None:
        _detector = PoseDetector.from_config(load_config())

    _detector.load_model()


@app.post("/camera/start")
async def start_camera(
    source: int = 0,
    patient_id: str = "Patient 01",
    room_id: str = "ICU-101",
) -> dict[str, Any]:
    global _camera_thread, _camera_stop_event, _camera_status

    if _camera_thread is not None and _camera_thread.is_alive():
        return {"ok": True, "message": "Camera is already running"}

    try:
        await run_in_threadpool(_initialize_detector)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not initialize YOLO: {exc}",
        ) from exc

    monitor = get_monitor()
    monitor.patient_id = patient_id
    monitor.room_id = room_id

    _camera_stop_event = threading.Event()
    loop = asyncio.get_running_loop()

    _camera_status = {
        "running": True,
        "frames_processed": 0,
        "last_error": None,
    }

    _camera_thread = threading.Thread(
        target=_camera_worker,
        args=(source, _camera_stop_event, loop),
        daemon=True,
    )
    _camera_thread.start()

    return {"ok": True, "message": "Camera starting"}


@app.post("/camera/stop")
async def stop_camera() -> dict[str, Any]:
    global _camera_status

    if _camera_stop_event is not None:
        _camera_stop_event.set()

    if _camera_thread is not None and _camera_thread.is_alive():
        await run_in_threadpool(_camera_thread.join, 5)

    return {"ok": True, "camera": _camera_status}


@app.get("/camera/status")
def camera_status() -> dict[str, Any]:
    return _camera_status
@app.post("/predict/video", response_model=PredictResponse)
def predict_video(body: VideoPredictRequest) -> PredictResponse:
    path = resolve_path(body.video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {path}")

    global _detector
    cfg = load_config()
    if _detector is None:
        try:
            _detector = PoseDetector.from_config(cfg)
            _detector.load_model()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"YOLO Pose failed: {exc}") from exc

    mon = FrameMonitor(
        cfg,
        patient_id=body.patient_id,
        room_id=body.room_id,
    )
    last: dict[str, Any] | None = None
    n = 0
    try:
        for frame, pose in _detector.process_video(
            path,
            max_frames=body.max_frames,
            frame_skip=body.frame_skip,
            display=False,
        ):
            last = mon.process_pose_frame(pose, frame_shape=frame.shape)
            n += 1
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(
        ok=True,
        result={
            "frames_processed": n,
            "last": last,
            "events": mon.events.all(),
            "lstm_available": mon.lstm.available,
        },
    )


@app.websocket("/ws/monitor")
async def ws_monitor(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.append(ws)
    try:
        mon = get_monitor()
        await ws.send_json(
            {
                "type": "snapshot",
                "risk": None if mon.last_risk is None else mon.last_risk.to_dict(),
                "events": mon.events.latest(10),
            }
        )
        while True:
            await ws.receive_text()
            await ws.send_json(
                {
                    "type": "snapshot",
                    "risk": None if mon.last_risk is None else mon.last_risk.to_dict(),
                    "events": mon.events.latest(10),
                }
            )
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
