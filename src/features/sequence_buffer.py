"""Rolling temporal buffers for per-person keypoint sequences."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceReady:
    person_id: int
    sequence: np.ndarray  # (T, F)
    frame_number: int
    timestamp: float


class RollingSequenceBuffer:
    """Sliding window of length ``sequence_length``, emit every ``stride`` frames."""

    def __init__(
        self,
        sequence_length: int = 20,
        stride: int = 5,
        feature_dim: int = 99,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_dim = feature_dim
        self._frames: deque[np.ndarray] = deque(maxlen=sequence_length)
        self._since_emit = 0
        self._filled_once = False

    def reset(self) -> None:
        self._frames.clear()
        self._since_emit = 0
        self._filled_once = False

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def is_full(self) -> bool:
        return len(self._frames) >= self.sequence_length

    def push(
        self,
        features: np.ndarray,
        person_id: int,
        frame_number: int,
        timestamp: float,
    ) -> SequenceReady | None:
        vec = np.asarray(features, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {vec.shape[0]}"
            )
        self._frames.append(vec)
        if not self.is_full:
            return None

        if not self._filled_once:
            self._filled_once = True
            self._since_emit = 0
            return SequenceReady(
                person_id=person_id,
                sequence=np.stack(self._frames, axis=0),
                frame_number=frame_number,
                timestamp=timestamp,
            )

        self._since_emit += 1
        if self._since_emit < self.stride:
            return None
        self._since_emit = 0
        return SequenceReady(
            person_id=person_id,
            sequence=np.stack(self._frames, axis=0),
            frame_number=frame_number,
            timestamp=timestamp,
        )


class PersonSequenceStore:
    """One rolling buffer per tracked person ID."""

    def __init__(self, sequence_length: int = 20, stride: int = 5, feature_dim: int = 99) -> None:
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_dim = feature_dim
        self._buffers: dict[int, RollingSequenceBuffer] = {}

    def push(
        self,
        person_id: int,
        features: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> SequenceReady | None:
        buf = self._buffers.get(person_id)
        if buf is None:
            buf = RollingSequenceBuffer(
                self.sequence_length, self.stride, self.feature_dim
            )
            self._buffers[person_id] = buf
        return buf.push(features, person_id, frame_number, timestamp)

    def drop(self, person_id: int) -> None:
        self._buffers.pop(person_id, None)
