#!/usr/bin/env python3
"""Demo: RAW VIDEO / WEBCAM → YOLO Pose → skeleton visualization.

Examples
--------
    python demo_pose.py --source path/to/video.mp4
    python demo_pose.py --source 0
    python demo_pose.py --source path/to/video.mp4 --no-display --save-keypoints out.jsonl
    python demo_pose.py --image path/to/person.jpg

Privacy: by default this demo does not write raw video. Optional keypoint
JSONL stores pose metadata only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

# Allow running from repository root without installing the package
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose.pose_detector import PoseDetector
from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging

logger = logging.getLogger("demo_pose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PatientSafetyAI YOLO Pose demo (CPU-friendly)"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video path or webcam index (e.g. 0). Ignored if --image is set.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Single image path for a quick pose smoke test.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override YOLO pose weights path (default from config).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (default from config, usually cpu).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Skip N frames between processed frames.",
    )
    parser.add_argument(
        "--save-keypoints",
        default=None,
        help="Write keypoints JSONL (no raw video).",
    )
    parser.add_argument(
        "--save-vis",
        default=None,
        help="Optional path to write an annotated video (privacy trade-off).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window.",
    )
    parser.add_argument(
        "--person-id",
        type=int,
        default=None,
        help="Highlight a specific tracked person ID when present.",
    )
    return parser.parse_args()


def build_detector(args: argparse.Namespace) -> PoseDetector:
    config = load_config()
    detector = PoseDetector.from_config(config)
    if args.model:
        detector.model_path = Path(args.model)
        if not detector.model_path.is_absolute():
            detector.model_path = ROOT / detector.model_path
        detector._model = None
    if args.device:
        detector.device = args.device
    detector.load_model()
    return detector


def run_image(detector: PoseDetector, image_path: Path, display: bool) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("Could not read image: %s", image_path)
        return 1

    pose = detector.process_frame(frame, frame_number=0, timestamp=0.0)
    logger.info(
        "Image %s | persons=%s",
        image_path.name,
        pose.person_count,
    )
    for person in pose.persons:
        logger.info(
            "  person_id=%s det_conf=%.3f mean_kpt_conf=%.3f",
            person.person_id,
            person.detection_confidence,
            person.mean_keypoint_confidence(),
        )

    vis = detector.draw_skeleton(
        frame,
        pose,
        overlay_text=[
            "PatientSafetyAI Pose Demo",
            f"Persons: {pose.person_count}",
            "Format: COCO-17 (not UP-Fall 99)",
        ],
    )
    out_path = ROOT / "results" / "pose_demo_image.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    logger.info("Saved visualization to %s", out_path)

    if display:
        cv2.imshow("PatientSafetyAI Pose", vis)
        logger.info("Press any key in the window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_stream(detector: PoseDetector, args: argparse.Namespace) -> int:
    source = args.source
    if source is None:
        logger.error("Provide --source (video path or webcam index) or --image")
        return 1

    display = not args.no_display
    is_webcam = str(source).isdigit()

    writer = None
    kp_file = None
    if args.save_keypoints:
        kp_path = Path(args.save_keypoints)
        kp_path.parent.mkdir(parents=True, exist_ok=True)
        kp_file = kp_path.open("w", encoding="utf-8")

    try:
        if is_webcam:
            stream = detector.process_webcam(
                source=int(source),
                max_frames=args.max_frames,
                display=False,
            )
            label = f"webcam:{source}"
        else:
            video_path = Path(source)
            if not video_path.exists():
                logger.error("Video not found: %s", video_path)
                return 1
            stream = detector.process_video(
                video_path,
                max_frames=args.max_frames,
                frame_skip=args.frame_skip,
                display=False,
            )
            label = video_path.name

        frame_count = 0
        person_frames = 0
        t0 = time.perf_counter()

        for frame, pose in stream:
            frame_count += 1
            if pose.person_count > 0:
                person_frames += 1

            selected = pose.select_person(person_id=args.person_id)
            fps = frame_count / max(time.perf_counter() - t0, 1e-6)

            hud = [
                "PatientSafetyAI | Pose Demo",
                f"Source: {label}",
                f"Persons: {pose.person_count}",
                f"FPS: {fps:.1f}",
            ]
            if selected is not None:
                hud.append(
                    f"Track ID {selected.person_id} "
                    f"conf={selected.detection_confidence:.2f}"
                )
            hud.append("Keypoints: COCO-17 (YOLO)")
            hud.append("LSTM bridge: not connected yet")

            vis = detector.draw_skeleton(frame, pose, overlay_text=hud)

            if kp_file is not None:
                kp_file.write(json.dumps(pose.to_dict()) + "\n")

            if args.save_vis:
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_path = Path(args.save_vis)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (w, h))
                    logger.warning(
                        "Saving annotated video to %s — this stores "
                        "identifiable imagery. Prefer --save-keypoints for "
                        "privacy-conscious logging.",
                        out_path,
                    )
                writer.write(vis)

            if display:
                cv2.imshow("PatientSafetyAI Pose", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Stopped by user")
                    break

            if frame_count % 30 == 0:
                logger.info(
                    "frame=%s persons=%s fps=%.1f",
                    pose.frame_number,
                    pose.person_count,
                    fps,
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Done | frames=%s person_frames=%s elapsed=%.1fs avg_fps=%.2f",
            frame_count,
            person_frames,
            elapsed,
            frame_count / max(elapsed, 1e-6),
        )
        return 0
    finally:
        if writer is not None:
            writer.release()
        if kp_file is not None:
            kp_file.close()
        if display:
            cv2.destroyAllWindows()


def main() -> int:
    setup_logging()
    args = parse_args()
    try:
        detector = build_detector(args)
    except Exception as exc:
        logger.error("Failed to load pose model: %s", exc)
        return 1

    if args.image:
        return run_image(detector, Path(args.image), display=not args.no_display)
    return run_stream(detector, args)


if __name__ == "__main__":
    raise SystemExit(main())
