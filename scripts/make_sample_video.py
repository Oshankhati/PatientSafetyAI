# Create a short synthetic MP4 with a moving "person-like" rectangle.
# Used only when no real video is available; YOLO may not detect it.
# Prefer downloading a real person image for smoke tests.

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "samples"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    out_video = OUT / "synthetic_motion.mp4"
    out_img = OUT / "synthetic_frame.jpg"
    w, h, fps, n = 640, 480, 15, 45
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for i in range(n):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        x = 80 + i * 8
        y = 120 + int(20 * np.sin(i / 4))
        # crude stick-figure-ish blob (not guaranteed detectable by YOLO)
        cv2.rectangle(frame, (x, y), (x + 80, y + 220), (180, 180, 200), -1)
        cv2.circle(frame, (x + 40, y - 20), 25, (200, 180, 160), -1)
        writer.write(frame)
        if i == n // 2:
            cv2.imwrite(str(out_img), frame)
    writer.release()
    print(f"Wrote {out_video}")
    print(f"Wrote {out_img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
