"""JSON configuration: machines, lamp ROIs and thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ROI = tuple[int, int, int, int]  # x, y, w, h in pixels
COLORS = ("red", "amber", "green")


@dataclass(frozen=True)
class Thresholds:
    """Tunable detection and timing parameters."""

    sat_min: float = 0.40  # min saturation for a pixel to count as lamp-coloured
    lit_ratio: float = 1.8  # lit pixel V must exceed this x the lamp's own "off" level
    min_lit_v: float = 0.35  # absolute floor for a lit pixel (rejects noise in the dark)
    init_lit_v: float = 0.60  # lit-pixel V used until an "off" baseline has been learned
    on_fraction: float = 0.20  # share of the ROI that must be lit for the lamp to be ON
    baseline_alpha: float = 0.02  # EMA rate of the off-baseline (updated only while OFF)
    window_s: float = 2.0  # sliding window for blink analysis
    settle_s: float = 0.3  # frames this recent must agree to call steady/off
    min_blink_edges: int = 3  # on/off transitions needed inside the window
    max_gap_s: float = 0.7  # max time between transitions of one blink run
    blink_min_hz: float = 0.7
    blink_max_hz: float = 4.0
    min_dwell_s: float = 1.0  # a new machine state must persist this long


@dataclass(frozen=True)
class LampCfg:
    color: str
    roi: ROI


@dataclass(frozen=True)
class MachineCfg:
    name: str
    lamps: tuple[LampCfg, ...]


@dataclass(frozen=True)
class Config:
    line: str
    fps: float
    machines: tuple[MachineCfg, ...]
    thresholds: Thresholds = field(default_factory=Thresholds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "fps": self.fps,
            "thresholds": {f.name: getattr(self.thresholds, f.name) for f in fields(Thresholds)},
            "machines": [
                {"name": m.name, "lamps": [{"color": lamp.color, "roi": list(lamp.roi)} for lamp in m.lamps]}
                for m in self.machines
            ],
        }


def config_from_dict(data: dict[str, Any]) -> Config:
    """Validate and build a Config from parsed JSON."""
    known = {f.name for f in fields(Thresholds)}
    th_raw = data.get("thresholds", {})
    unknown = set(th_raw) - known
    if unknown:
        raise ValueError(f"unknown threshold keys: {sorted(unknown)}")
    fps = float(data.get("fps", 10))
    if fps <= 0:
        raise ValueError("fps must be > 0")
    machines: list[MachineCfg] = []
    for m in data.get("machines", []):
        lamps = []
        for lamp in m.get("lamps", []):
            color = lamp["color"]
            if color not in COLORS:
                raise ValueError(f"lamp colour must be one of {COLORS}, got {color!r}")
            roi = tuple(int(v) for v in lamp["roi"])
            if len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0 or roi[0] < 0 or roi[1] < 0:
                raise ValueError(f"bad ROI {lamp['roi']!r} (need x, y >= 0 and w, h > 0)")
            lamps.append(LampCfg(color, roi))  # type: ignore[arg-type]
        if not lamps or len({lamp.color for lamp in lamps}) != len(lamps):
            raise ValueError(f"machine {m.get('name')!r}: need at least one lamp, one per colour")
        machines.append(MachineCfg(str(m["name"]), tuple(lamps)))
    if not machines:
        raise ValueError("config needs at least one machine")
    if len({m.name for m in machines}) != len(machines):
        raise ValueError("machine names must be unique")
    return Config(str(data.get("line", "line-1")), fps, tuple(machines), Thresholds(**th_raw))


def load_config(path: str | Path) -> Config:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
