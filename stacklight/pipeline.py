"""Frame -> events pipeline for a whole line."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .config import Config
from .detect import LampDetector, scene_gain
from .oee import OeeReport, compute_oee, intervals_from_events
from .state import MachineStateMachine, State, StateEvent, classify
from .temporal import LampMode, LampTracker


class LineMonitor:
    """Runs detect -> temporal -> state for every machine in the config.

    The first `window_s` of video is a warm-up that fills the blink windows; the initial
    state it yields is back-dated to the first frame.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        th = cfg.thresholds
        self._lamps = {
            m.name: [(lamp.color, LampDetector(lamp.color, lamp.roi, th), LampTracker(cfg.fps, th))
                     for lamp in m.lamps]
            for m in cfg.machines
        }
        self._machines: dict[str, MachineStateMachine] = {}
        self._warmup = next(iter(self._lamps.values()))[0][2].size
        self._seen = 0
        self._t0: float | None = None
        self.events: list[StateEvent] = []

    @property
    def states(self) -> dict[str, State | None]:
        return {m.name: self._machines[m.name].state if m.name in self._machines else None
                for m in self.cfg.machines}

    @property
    def initial_states(self) -> dict[str, State]:
        return {n: sm.initial_state for n, sm in self._machines.items() if sm.initial_state}

    def process(self, frame: np.ndarray, ts: float) -> list[StateEvent]:
        if self._t0 is None:
            self._t0 = ts
        gain = scene_gain(frame)
        self._seen += 1
        new: list[StateEvent] = []
        for name, lamps in self._lamps.items():
            modes: dict[str, LampMode] = {
                color: tracker.update(det.update(frame, gain).on) for color, det, tracker in lamps
            }
            blink_ages = [tr.blink_age_s for _, _, tr in lamps if tr.mode is LampMode.BLINK]
            since = ts - max(blink_ages) if blink_ages else None
            if self._seen < self._warmup:
                continue
            sm = self._machines.setdefault(
                name, MachineStateMachine(name, self.cfg.thresholds.min_dwell_s, self._t0))
            ev = sm.update(ts, classify(modes), since)
            if ev:
                new.append(ev)
        self.events.extend(new)
        return new

    def oee(self, t_end: float, include_off: bool = False) -> list[OeeReport]:
        """Availability report per machine from the start of the video to `t_end`."""
        reports = []
        for name, sm in self._machines.items():
            evs = [e for e in self.events if e.machine == name]
            ivs = intervals_from_events(evs, sm.initial_state, self._t0 or 0.0, t_end)  # type: ignore[arg-type]
            reports.append(compute_oee(name, ivs, include_off))
        return reports


def run_frames(monitor: LineMonitor, frames: Iterable[tuple[float, np.ndarray]],
               on_event: Callable[[StateEvent], None] | None = None) -> float:
    """Feed (timestamp, frame) pairs through the monitor. Returns the last timestamp seen."""
    last = 0.0
    for ts, frame in frames:
        last = ts
        for ev in monitor.process(frame, ts):
            if on_event:
                on_event(ev)
    return last
