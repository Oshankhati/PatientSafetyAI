"""YOLO Pose detector wrapper for video and webcam input.

Designed for CPU-first academic demos. Prefer lightweight weights such as
``yolo11n-pose.pt``. Tracking is optional and used to keep person IDs stable
when multiple people appear in the frame.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.pose.keypoint_extractor import KeypointExtractor
from src.pose.schemas import FramePoseResult, PersonPose
from src.pose.skeleton_visualizer import SkeletonVisualizer
from src.utils.paths import project_root, resolve_path

logger = logging.getLogger(__name__)


class PoseDetector:
    """High-level pose estimation interface around Ultralytics YOLO Pose."""

    def __init__(
        self,
        model_path: str | Path = "models/yolo11n-pose.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        keypoint_conf_threshold: float = 0.30,
        imgsz: int = 640,
        enable_tracking: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model_path = resolve_path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.imgsz = imgsz
        self.enable_tracking = enable_tracking
        self.tracker = tracker

        self._model: Any | None = None
        self.extractor = KeypointExtractor(keypoint_conf_threshold)
        self.visualizer = SkeletonVisualizer(keypoint_conf_threshold)

    def load_model(self) -> None:
        """Load the YOLO Pose weights. Safe to call multiple times."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO Pose model not found at {self.model_path}. "
                "Download with: from ultralytics import YOLO; "
                "YOLO('yolo11n-pose.pt') and place it under models/."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for pose detection. "
                "Install with: pip install ultralytics"
            ) from exc

        logger.info("Loading YOLO Pose model from %s (device=%s)", self.model_path, self.device)
        self._model = YOLO(str(self.model_path))
        logger.info("YOLO Pose model loaded")

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load_model()
        return self._model

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
        persist_tracking: bool = False,
    ) -> FramePoseResult:
        """Run pose estimation on a single BGR frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: expected a non-empty numpy BGR image")

        predict_kwargs: dict[str, Any] = {
            "source": frame,
            "device": self.device,
            "conf": self.conf_threshold,
            "imgsz": self.imgsz,
            "verbose": False,
        }

        try:
            if self.enable_tracking and persist_tracking:
                try:
                    results = self.model.track(
                        persist=True,
                        tracker=self.tracker,
                        **predict_kwargs,
                    )
                except Exception as track_exc:
                    # Tracking deps (e.g. lap) may be missing; fall back to predict.
                    logger.warning(
                        "Tracking unavailable (%s); falling back to detect-only mode",
                        track_exc,
                    )
                    self.enable_tracking = False
                    results = self.model.predict(**predict_kwargs)
            else:
                results = self.model.predict(**predict_kwargs)
        except Exception as exc:
            logger.error("YOLO inference failed on frame %s: %s", frame_number, exc)
            raise RuntimeError(f"YOLO Pose inference failed: {exc}") from exc

        result = results[0]
        return self.extractor.extract(
            result,
            frame_number=frame_number,
            timestamp=timestamp,
            frame_shape=frame.shape,
        )

    def extract_keypoints(self, result: FramePoseResult) -> list[PersonPose]:
        """Return person poses from a frame result (API convenience)."""
        return list(result.persons)

    def draw_skeleton(
        self,
        frame: np.ndarray,
        result: FramePoseResult,
        overlay_text: list[str] | None = None,
    ) -> np.ndarray:
        """Draw boxes + skeletons on a copy of the frame."""
        canvas = frame.copy()
        return self.visualizer.draw_frame(canvas, result, overlay_text=overlay_text)

    def process_video(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
        frame_skip: int = 0,
        save_keypoints_path: str | Path | None = None,
        display: bool = False,
        window_name: str = "PatientSafetyAI Pose",
    ) -> Generator[tuple[np.ndarray, FramePoseResult], None, None]:
        """Yield (frame, pose_result) for each processed frame of a video file.

        Parameters
        ----------
        frame_skip:
            Number of frames to skip after each processed frame (0 = every frame).
        save_keypoints_path:
            Optional JSONL path for privacy-conscious keypoint logging (no raw video).
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        writer = None
        if save_keypoints_path is not None:
            out_path = Path(save_keypoints_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = out_path.open("w", encoding="utf-8")

        frame_idx = 0
        processed = 0
        t0 = time.perf_counter()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_skip > 0 and (frame_idx % (frame_skip + 1)) != 0:
                    frame_idx += 1
                    continue

                timestamp = frame_idx / fps if fps > 0 else float(frame_idx)
                pose = self.process_frame(
                    frame,
                    frame_number=frame_idx,
                    timestamp=timestamp,
                    persist_tracking=True,
                )
                pose.fps_hint = processed / max(time.perf_counter() - t0, 1e-6)

                if writer is not None:
                    writer.write(json.dumps(pose.to_dict()) + "\n")

                if display:
                    hud = [
                        f"Frame {frame_idx}",
                        f"Persons {pose.person_count}",
                        f"FPS {pose.fps_hint:.1f}" if pose.fps_hint else "",
                    ]
                    vis = self.draw_skeleton(frame, pose, overlay_text=[h for h in hud if h])
                    cv2.imshow(window_name, vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Display interrupted by user")
                        break

                yield frame, pose
                processed += 1
                frame_idx += 1

                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.close()
            if display:
                cv2.destroyWindow(window_name)

        logger.info(
            "Processed %s frames from %s (avg FPS ~%.2f)",
            processed,
            path.name,
            processed / max(time.perf_counter() - t0, 1e-6),
        )

    def process_webcam(
        self,
        source: int | str = 0,
        max_frames: int | None = None,
        display: bool = True,
        window_name: str = "PatientSafetyAI Webcam Pose",
    ) -> Generator[tuple[np.ndarray, FramePoseResult], None, None]:
        """Yield frames from a webcam / camera index."""
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Camera source unavailable: {source}. "
                "Check that a webcam is connected and not used by another app."
            )

        frame_idx = 0
        t0 = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Failed to read frame from camera %s", source)
                    break

                timestamp = time.time()
                pose = self.process_frame(
                    frame,
                    frame_number=frame_idx,
                    timestamp=timestamp,
                    persist_tracking=True,
                )
                pose.fps_hint = frame_idx / max(time.perf_counter() - t0, 1e-6)

                if display:
                    hud = [
                        f"Webcam frame {frame_idx}",
                        f"Persons {pose.person_count}",
                        f"FPS {pose.fps_hint:.1f}",
                    ]
                    vis = self.draw_skeleton(frame, pose, overlay_text=hud)
                    cv2.imshow(window_name, vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                yield frame, pose
                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
        finally:
            cap.release()
            if display:
                cv2.destroyWindow(window_name)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> PoseDetector:
        """Build a detector from ``config/config.yaml`` pose section."""
        if config is None:
            from src.utils.config_loader import load_config

            config = load_config()
        pose_cfg = config.get("pose", {})
        return cls(
            model_path=pose_cfg.get("model_path", "models/yolo11n-pose.pt"),
            device=str(pose_cfg.get("device", "cpu")),
            conf_threshold=float(pose_cfg.get("conf_threshold", 0.25)),
            keypoint_conf_threshold=float(
                pose_cfg.get("keypoint_conf_threshold", 0.30)
            ),
            imgsz=int(pose_cfg.get("imgsz", 640)),
            enable_tracking=bool(pose_cfg.get("enable_tracking", True)),
            tracker=str(pose_cfg.get("tracker", "bytetrack.yaml")),
        )


def save_keypoints_jsonl(
    results: Iterable[FramePoseResult],
    output_path: str | Path,
) -> Path:
    """Persist frame pose metadata without storing raw video frames."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item.to_dict()) + "\n")
    return path
