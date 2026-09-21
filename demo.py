#!/usr/bin/env python3
"""Full pipeline demo: video/webcam → YOLO Pose → sequences → risk.

Examples
--------
    python demo.py --source datasets/samples/bus_clip.mp4 --no-display --max-frames 15
    python demo.py --source 0
    python demo.py --image datasets/samples/bus.jpg --no-display

Does not store raw video by default. Optional --save-keypoints writes pose JSONL.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.monitor import FrameMonitor
from src.pose.pose_detector import PoseDetector
from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging

logger = logging.getLogger("demo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatientSafetyAI end-to-end pose/risk demo")
    p.add_argument("--source", default=None, help="Video path or webcam index")
    p.add_argument("--image", default=None, help="Single image path")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--frame-skip", type=int, default=0)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save-keypoints", default=None)
    p.add_argument("--person-id", type=int, default=None)
    p.add_argument("--patient", default="Patient 01")
    p.add_argument("--room", default="ICU-101")
    p.add_argument(
        "--publish-api",
        default=None,
        help="POST keypoints to FastAPI /predict/frame so the dashboard updates "
        "(example: http://127.0.0.1:8000). Sends skeleton metadata, not raw video.",
    )
    return p.parse_args()


def _publish(api_base: str, pose, patient_id: str, room_id: str) -> None:
    if not pose.persons:
        return
    body = {
        "frame_number": pose.frame_number,
        "timestamp": pose.timestamp,
        "patient_id": patient_id,
        "room_id": room_id,
        "persons": [
            {
                "person_id": p.person_id,
                "keypoints_normalized": p.keypoints_normalized.tolist(),
                "keypoint_confidence": p.keypoint_confidence.tolist(),
                "bbox": list(p.bounding_box.as_xyxy),
                "detection_confidence": p.detection_confidence,
            }
            for p in pose.persons
        ],
    }
    req = urllib.request.Request(
        api_base.rstrip("/") + "/predict/frame",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("API publish failed: %s", exc)


def hud_lines(status: dict, fps: float) -> list[str]:
    risk = status.get("risk") or {}
    return [
        "PatientSafetyAI | prototype (not clinical)",
        f"Patient: {status.get('patient_id', '-')}  Room: {status.get('room_id', '-')}",
        f"Persons: {status.get('person_count', 0)}  FPS: {fps:.1f}",
        f"Prediction: {risk.get('prediction_label', '—')}",
        f"Risk: {risk.get('risk_level', '—')}  Score: {risk.get('risk_score', '—')}",
        f"LSTM p(fall): {risk.get('lstm_fall_probability', 'n/a')}  zone={status.get('zone', '-')}",
    ]


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = load_config()
    try:
        detector = PoseDetector.from_config(cfg)
        detector.load_model()
    except Exception as exc:
        logger.error("Failed to load YOLO Pose: %s", exc)
        return 1

    monitor = FrameMonitor(
        cfg,
        patient_id=args.patient,
        room_id=args.room,
        target_person_id=args.person_id,
    )
    if not monitor.lstm.available:
        logger.warning("LSTM unavailable — using pose signals only. %s", monitor.lstm.load_error)

    display = not args.no_display
    kp_file = None
    if args.save_keypoints:
        path = Path(args.save_keypoints)
        path.parent.mkdir(parents=True, exist_ok=True)
        kp_file = path.open("w", encoding="utf-8")

    try:
        if args.image:
            frame = cv2.imread(str(Path(args.image)))
            if frame is None:
                logger.error("Could not read image %s", args.image)
                return 1
            pose = detector.process_frame(frame, 0, 0.0)
            status = monitor.process_pose_frame(pose, frame.shape)
            if args.publish_api:
                _publish(args.publish_api, pose, args.patient, args.room)
            vis = detector.draw_skeleton(frame, pose, overlay_text=hud_lines(status, 0.0))
            out = ROOT / "results" / "demo_image.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis)
            logger.info("Saved %s | %s", out, status.get("risk"))
            if display:
                cv2.imshow("PatientSafetyAI", vis)
                cv2.waitKey(0)
            return 0

        if args.source is None:
            logger.error("Provide --source or --image")
            return 1

        is_cam = str(args.source).isdigit()
        if is_cam:
            stream = detector.process_webcam(int(args.source), args.max_frames, display=False)
        else:
            video = Path(args.source)
            if not video.exists():
                logger.error("Video not found: %s", video)
                return 1
            stream = detector.process_video(
                video,
                max_frames=args.max_frames,
                frame_skip=args.frame_skip,
                display=False,
            )

        t0 = time.perf_counter()
        n = 0
        for frame, pose in stream:
            n += 1
            status = monitor.process_pose_frame(pose, frame.shape)
            if args.publish_api:
                _publish(args.publish_api, pose, args.patient, args.room)
            fps = n / max(time.perf_counter() - t0, 1e-6)
            vis = detector.draw_skeleton(frame, pose, overlay_text=hud_lines(status, fps))
            if kp_file is not None:
                rec = pose.to_dict()
                rec["risk"] = status.get("risk")
                kp_file.write(json.dumps(rec) + "\n")
            if display:
                cv2.imshow("PatientSafetyAI", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n % 10 == 0:
                logger.info("frame=%s risk=%s", pose.frame_number, status.get("risk"))

        logger.info("Processed %s frames | events=%s", n, len(monitor.events.all()))
        return 0
    finally:
        if kp_file is not None:
            kp_file.close()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
