"""Per-ROI lamp detection with a vectorised numpy RGB->HSV and a rolling off-baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ROI, Thresholds

# (centre hue, half width) in degrees
HUE_BANDS: dict[str, tuple[float, float]] = {
    "red": (0.0, 22.0),
    "amber": (40.0, 16.0),
    "green": (125.0, 38.0),
}


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (..., 3) RGB to (h in [0, 360), s, v in [0, 1]). uint8 input is scaled by 1/255."""
    arr = np.asarray(rgb)
    x = arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else arr.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(axis=-1)
    c = v - x.min(axis=-1)
    has_c = c > 0
    safe_c = np.where(has_c, c, 1.0)
    is_r = has_c & (v == r)
    is_g = has_c & ~is_r & (v == g)
    is_b = has_c & ~is_r & ~is_g
    sector = np.zeros_like(v)
    sector = np.where(is_r, ((g - b) / safe_c) % 6.0, sector)
    sector = np.where(is_g, (b - r) / safe_c + 2.0, sector)
    sector = np.where(is_b, (r - g) / safe_c + 4.0, sector)
    h = (sector * 60.0) % 360.0
    s = np.where(v > 0, c / np.where(v > 0, v, 1.0), 0.0)
    return h, s.astype(np.float32), v


def hue_in_band(h: np.ndarray, color: str) -> np.ndarray:
    centre, half = HUE_BANDS[color]
    return np.abs((h - centre + 180.0) % 360.0 - 180.0) <= half


def scene_gain(frame: np.ndarray) -> float:
    """Global brightness reference: median V of a subsampled frame (never below 0.02)."""
    v = frame[::4, ::4].max(axis=-1).astype(np.float32)
    if frame.dtype == np.uint8:
        v /= 255.0
    return max(float(np.median(v)), 0.02)


@dataclass(frozen=True)
class LampReading:
    on: bool
    lit_fraction: float
    baseline: float | None


class LampDetector:
    """Decides ON/OFF for one lamp ROI.

    A pixel is lit when its hue is in the lamp's band, it is saturated, and its brightness
    exceeds `lit_ratio` x the lamp's own recent OFF brightness. That OFF level is stored
    relative to the scene gain, so a slow day/night drift or a sudden light switch moves
    the threshold with the scene. The baseline only learns while the lamp is OFF, so a lamp
    that stays on for hours is not "learned away".
    """

    def __init__(self, color: str, roi: ROI, th: Thresholds) -> None:
        if color not in HUE_BANDS:
            raise ValueError(f"unknown lamp colour {color!r}")
        self.color, self.roi, self.th = color, roi, th
        self.baseline: float | None = None  # off-level V / scene gain

    def update(self, frame: np.ndarray, gain: float) -> LampReading:
        x, y, w, h = self.roi
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            raise ValueError(f"ROI {self.roi} is outside the frame {frame.shape[:2]}")
        hue, sat, val = rgb_to_hsv(crop)
        matched = hue_in_band(hue, self.color) & (sat >= self.th.sat_min)
        if self.baseline is None:
            v_thr = self.th.init_lit_v
        else:
            v_thr = max(self.th.min_lit_v, self.th.lit_ratio * self.baseline * gain)
        lit_fraction = float((matched & (val >= v_thr)).mean())
        on = lit_fraction >= self.th.on_fraction
        if not on and matched.mean() >= 0.2:
            observed = float(np.median(val[matched])) / gain
            if self.baseline is None:
                self.baseline = observed
            else:
                self.baseline += self.th.baseline_alpha * (observed - self.baseline)
        return LampReading(on, lit_fraction, self.baseline)
