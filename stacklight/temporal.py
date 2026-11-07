"""Temporal analysis of one lamp's on/off stream: flicker filtering and blink detection."""

from __future__ import annotations

from collections import deque
from enum import Enum

from .config import Thresholds


class LampMode(str, Enum):
    OFF = "off"
    STEADY = "steady"
    BLINK = "blink"


def _median3(x: list[bool]) -> list[bool]:
    """Remove single-frame dropouts/spikes; the newest sample is left as-is."""
    out = list(x)
    for i in range(1, len(x) - 1):
        out[i] = (x[i - 1] + x[i] + x[i + 1]) >= 2
    return out


class LampTracker:
    """Sliding-window classifier: OFF, STEADY or BLINK.

    BLINK needs `min_blink_edges` transitions in the window, spaced at most `max_gap_s`
    apart (the last one included), at a rate inside [blink_min_hz, blink_max_hz].
    Otherwise the last `settle_s` of frames decide STEADY/OFF; if they disagree the
    previous mode is kept.
    """

    def __init__(self, fps: float, th: Thresholds) -> None:
        self.fps, self.th = fps, th
        self.size = max(4, round(th.window_s * fps))
        self._settle = max(2, round(th.settle_s * fps))
        self._buf: deque[bool] = deque(maxlen=self.size)
        self.mode = LampMode.OFF
        self.blink_age_s = 0.0  # while BLINK: how long ago the current blink run began

    def update(self, on: bool) -> LampMode:
        self._buf.append(bool(on))
        arr = _median3(list(self._buf))
        if len(arr) < 3:
            self.mode = LampMode.STEADY if arr[-1] else LampMode.OFF
            return self.mode
        if self._is_blinking(arr):
            self.mode = LampMode.BLINK
            return self.mode
        recent = arr[-self._settle :]
        if all(recent):
            self.mode = LampMode.STEADY
        elif not any(recent):
            self.mode = LampMode.OFF
        return self.mode

    def _is_blinking(self, arr: list[bool]) -> bool:
        th = self.th
        edges = [i for i in range(1, len(arr)) if arr[i] != arr[i - 1]]
        if len(edges) < th.min_blink_edges:
            return False
        gaps = [b - a for a, b in zip(edges, edges[1:])] + [len(arr) - 1 - edges[-1]]
        if max(gaps) / self.fps > th.max_gap_s:
            return False
        span_s = (edges[-1] - edges[0]) / self.fps
        hz = (len(edges) - 1) / (2.0 * span_s)
        if not th.blink_min_hz <= hz <= th.blink_max_hz:
            return False
        self.blink_age_s = (len(arr) - 1 - edges[0]) / self.fps
        return True
