"""Live camera / RTSP input (optional: needs opencv-python)."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np


def iter_frames(source: str, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (seconds since start, RGB uint8 frame), throttled to roughly `fps`."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Camera input needs OpenCV. Install it with: pip install 'stacklight-oee-vision[camera]'"
        ) from exc
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")
    start, last = time.monotonic(), -1.0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            ts = time.monotonic() - start
            if last >= 0 and ts - last < 1.0 / fps:
                continue
            last = ts
            yield ts, np.ascontiguousarray(bgr[..., ::-1])
    finally:
        cap.release()
