"""Synthetic stack-light video: the test oracle.

Renders towers of red/amber/green lamps with sensor noise, lamp flicker, a slow
day/night brightness ramp and a sudden light switch. The scripted timeline is the
ground truth that the detection pipeline is scored against.
"""

from __future__ import annotations

import colorsys
import math
from bisect import bisect_right
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Iterator

import numpy as np

from .config import Config, LampCfg, MachineCfg
from .state import State, StateEvent

Timeline = list[tuple[State, float]]  # (state, seconds)
_R, _I, _S, _F, _O = State.RUNNING, State.IDLE, State.SETUP, State.FAULT, State.OFF

_TEMPLATES: list[Timeline] = [
    [(_R, 45), (_I, 12), (_R, 40), (_F, 18), (_R, 60), (_S, 20), (_R, 35), (_I, 10)],
    [(_S, 15), (_R, 70), (_F, 12), (_R, 50), (_I, 25), (_R, 45), (_O, 20), (_R, 30)],
    [(_R, 30), (_O, 25), (_R, 55), (_F, 15), (_R, 65), (_I, 18), (_R, 40), (_S, 22)],
]
# render hues in degrees, deliberately not equal to the detector's band centres
_LAMP_HUE = {"red": 358.0, "amber": 38.0, "green": 128.0}


@dataclass(frozen=True)
class MachineScript:
    name: str
    timeline: Timeline
    fault_blinks: bool = False  # red blinks instead of staying steady during FAULT


@dataclass(frozen=True)
class Scenario:
    scripts: tuple[MachineScript, ...]
    duration_s: float
    fps: float = 10.0
    seed: int = 7
    width: int = 320
    height: int = 240
    noise_sigma: float = 0.02
    flicker_prob: float = 0.01  # per lit lamp and frame: a one-frame dropout
    blink_hz: float = 1.5
    lights_off: tuple[float, float] = (0.55, 0.70)  # fraction of the run with room lights off
    lights_off_factor: float = 0.35
    extra: dict[str, float] = field(default_factory=dict)


def default_config(n_machines: int = 3, fps: float = 10.0, width: int = 320) -> Config:
    """Tower layout used by the simulator and by examples/line_config.json."""
    names = ["press-01", "lathe-02", "packer-03"] + [f"machine-{i:02d}" for i in range(4, n_machines + 1)]
    machines = []
    for i in range(n_machines):
        cx = int(width * (i + 0.5) / n_machines)
        lamps = tuple(LampCfg(c, (cx - 10, y, 20, 20)) for c, y in (("red", 40), ("amber", 80), ("green", 120)))
        machines.append(MachineCfg(names[i], lamps))
    return Config("line-1", fps, tuple(machines))


def _fit(timeline: Timeline, duration: float) -> Timeline:
    """Tile the template to `duration`; fold a too-short tail into the previous segment."""
    out: Timeline = []
    t = 0.0
    while t < duration:
        for state, secs in timeline:
            if t >= duration:
                break
            secs = min(secs, duration - t)
            if secs < 8 and out:
                out[-1] = (out[-1][0], out[-1][1] + secs)
            else:
                out.append((state, secs))
            t += secs
    return out


def default_scenario(cfg: Config, duration_s: float = 300.0, seed: int = 7) -> Scenario:
    scripts = tuple(
        MachineScript(m.name, _fit(_TEMPLATES[i % 3][i // 3 :] + _TEMPLATES[i % 3][: i // 3], duration_s),
                      fault_blinks=(i % 3 == 1))
        for i, m in enumerate(cfg.machines)
    )
    return Scenario(scripts, duration_s, fps=cfg.fps, seed=seed)


def truth_state(script: MachineScript, t: float) -> State:
    ends = list(accumulate(secs for _, secs in script.timeline))
    return script.timeline[min(bisect_right(ends, t), len(script.timeline) - 1)][0]


def truth_transitions(script: MachineScript) -> list[tuple[float, State]]:
    out, t = [], 0.0
    for i, (state, secs) in enumerate(script.timeline):
        if i > 0:
            out.append((t, state))
        t += secs
    return out


class Simulator:
    """Renders RGB uint8 frames of the lamps described by `cfg`, driven by `scenario`."""

    def __init__(self, cfg: Config, scenario: Scenario) -> None:
        if len(cfg.machines) != len(scenario.scripts):
            raise ValueError("scenario needs one script per configured machine")
        self.cfg, self.sc = cfg, scenario
        self.rng = np.random.default_rng(scenario.seed)
        self._phase = self.rng.random(len(cfg.machines))
        self._build_scene()

    def _build_scene(self) -> None:
        h, w = self.sc.height, self.sc.width
        base = 0.36 + 0.08 * np.linspace(0, 1, h, dtype=np.float32)[:, None]
        self._albedo = np.repeat(base[:, :, None], w, axis=1).repeat(3, axis=2)
        self._albedo += self.rng.normal(0, 0.015, (h, w, 1)).astype(np.float32)
        self._lamps: list[tuple[int, tuple[slice, slice], np.ndarray, np.ndarray, str]] = []
        for mi, m in enumerate(self.cfg.machines):
            xs = [r for lamp in m.lamps for r in (lamp.roi[0], lamp.roi[0] + lamp.roi[2])]
            ys = [r for lamp in m.lamps for r in (lamp.roi[1], lamp.roi[1] + lamp.roi[3])]
            self._albedo[min(ys) - 10 : max(ys) + 10, min(xs) - 6 : max(xs) + 6] = 0.10
            for lamp in m.lamps:
                x, y, rw, rh = lamp.roi
                sl = (slice(y, y + rh), slice(x, x + rw))
                yy, xx = np.mgrid[0:rh, 0:rw]
                r = np.hypot(yy - (rh - 1) / 2, xx - (rw - 1) / 2) / (min(rw, rh) / 2 - 0.5)
                disc = r <= 1.0
                pure = np.array(colorsys.hsv_to_rgb(_LAMP_HUE[lamp.color] / 360.0, 1.0, 1.0), dtype=np.float32)
                sat = np.clip(0.9 - 0.35 * (1 - r), 0, 1)[..., None]
                lit = 0.95 * ((1 - sat) + sat * pure)  # HSV(h, s, 0.95), whiter towards the centre
                off = 0.30 * (0.15 + 0.85 * pure)  # dull, tinted lens when unlit
                self._albedo[sl][disc] = off
                self._lamps.append((mi, sl, disc, lit.astype(np.float32), lamp.color))

    def ambient(self, t: float) -> float:
        d = self.sc.duration_s
        level = 0.75 + 0.25 * math.sin(2 * math.pi * t / d - math.pi / 2)  # slow day/night ramp
        lo, hi = self.sc.lights_off
        return level * (self.sc.lights_off_factor if lo * d <= t < hi * d else 1.0)

    def lamp_pattern(self, mi: int, t: float) -> dict[str, bool]:
        sc = self.sc.scripts[mi]
        state = truth_state(sc, t)
        blink_on = ((t * self.sc.blink_hz + self._phase[mi]) % 1.0) < 0.5
        return {
            "red": state is State.FAULT and (blink_on or not sc.fault_blinks),
            "amber": state is State.IDLE or (state is State.SETUP and blink_on),
            "green": state is State.RUNNING,
        }

    def render(self, t: float) -> np.ndarray:
        img = self._albedo * self.ambient(t)
        patterns = [self.lamp_pattern(mi, t) for mi in range(len(self.cfg.machines))]
        for mi, sl, disc, lit, color in self._lamps:
            if patterns[mi][color] and self.rng.random() >= self.sc.flicker_prob:
                gain = self.rng.uniform(0.85, 1.0)
                region = img[sl]
                region[disc] = np.maximum(region[disc], lit[disc] * gain)
        img *= 1.0 + self.rng.normal(0, 0.01)  # exposure jitter
        noise = self.rng.standard_normal(img.shape, dtype=np.float32)
        noise *= self.sc.noise_sigma
        img += noise
        np.clip(img, 0, 1, out=img)
        img *= 255
        img += 0.5
        return img.astype(np.uint8)

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        for i in range(int(self.sc.duration_s * self.sc.fps)):
            t = i / self.sc.fps
            yield t, self.render(t)


@dataclass(frozen=True)
class Evaluation:
    frames: int
    frame_accuracy: float
    transitions: int
    matched: int
    spurious: int
    max_time_error_s: float
    mean_time_error_s: float


def evaluate(scenario: Scenario, events: list[StateEvent], initial: dict[str, State],
             tol_s: float = 1.0) -> Evaluation:
    """Score detected events against the scripted timelines (per-frame labels and transitions)."""
    n_frames = int(scenario.duration_s * scenario.fps)
    correct = total = n_trans = matched = 0
    errors: list[float] = []
    spurious = 0
    for script in scenario.scripts:
        evs = [e for e in events if e.machine == script.name]
        times = [e.ts for e in evs]
        state = initial[script.name]
        k = 0
        for i in range(n_frames):
            t = i / scenario.fps
            while k < len(evs) and times[k] <= t:
                state, k = evs[k].to_state, k + 1
            correct += state is truth_state(script, t)
            total += 1
        used: set[int] = set()
        for t_true, to in truth_transitions(script):
            n_trans += 1
            cands = [(abs(e.ts - t_true), j) for j, e in enumerate(evs)
                     if j not in used and e.to_state is to and abs(e.ts - t_true) <= tol_s]
            if cands:
                err, j = min(cands)
                used.add(j)
                matched += 1
                errors.append(err)
        spurious += len(evs) - len(used)
    return Evaluation(
        frames=total, frame_accuracy=correct / total, transitions=n_trans, matched=matched,
        spurious=spurious, max_time_error_s=max(errors, default=0.0),
        mean_time_error_s=float(np.mean(errors)) if errors else 0.0,
    )
