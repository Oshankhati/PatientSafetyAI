"""Lightweight bounding-box tracker used when Ultralytics tracking is unavailable."""

from __future__ import annotations

from dataclasses import dataclass

from src.pose.schemas import BoundingBox


@dataclass
class _Track:
    box: BoundingBox
    last_frame: int
    previous_box: BoundingBox | None = None
    previous_frame: int | None = None


class IoUTracker:
    """Greedy IoU association with monotonically increasing, fallback-only IDs."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missing_frames: int = 30,
        id_start: int = 1_000_000,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._next_id = id_start
        self._tracks: dict[int, _Track] = {}

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
        area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
        return intersection / max(area_a + area_b - intersection, 1e-8)

    @staticmethod
    def _predict_box(track: _Track, frame_number: int) -> BoundingBox:
        if track.previous_box is None or track.previous_frame is None:
            return track.box
        elapsed = track.last_frame - track.previous_frame
        future = frame_number - track.last_frame
        if elapsed <= 0 or future <= 0:
            return track.box
        scale = future / elapsed
        previous = track.previous_box
        current = track.box
        return BoundingBox(
            current.x1 + (current.x1 - previous.x1) * scale,
            current.y1 + (current.y1 - previous.y1) * scale,
            current.x2 + (current.x2 - previous.x2) * scale,
            current.y2 + (current.y2 - previous.y2) * scale,
            current.confidence,
        )

    def update(self, boxes: list[BoundingBox], frame_number: int) -> list[int]:
        """Associate each box to a prior track, preserving detection order."""
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_number - track.last_frame > self.max_missing_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

        candidates = sorted(
            (
                (
                    max(
                        self._iou(track.box, box),
                        self._iou(self._predict_box(track, frame_number), box),
                    ),
                    track_id,
                    index,
                )
                for track_id, track in self._tracks.items()
                for index, box in enumerate(boxes)
            ),
            reverse=True,
        )
        assigned_tracks: set[int] = set()
        assigned_boxes: set[int] = set()
        box_ids: list[int | None] = [None] * len(boxes)

        for overlap, track_id, index in candidates:
            if overlap < self.iou_threshold:
                break
            if track_id in assigned_tracks or index in assigned_boxes:
                continue
            assigned_tracks.add(track_id)
            assigned_boxes.add(index)
            box_ids[index] = track_id

        for index, box in enumerate(boxes):
            track_id = box_ids[index]
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(box=box, last_frame=frame_number)
            else:
                previous = self._tracks[track_id]
                self._tracks[track_id] = _Track(
                    box=box,
                    last_frame=frame_number,
                    previous_box=previous.box,
                    previous_frame=previous.last_frame,
                )
            box_ids[index] = track_id

        return [int(track_id) for track_id in box_ids]
