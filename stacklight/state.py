"""Lamp pattern -> machine state, with a minimum-dwell state machine that emits events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .temporal import LampMode


class State(str, Enum):
    RUNNING = "RUNNING"  # green
    IDLE = "IDLE"  # amber steady (waiting / starved / blocked)
    SETUP = "SETUP"  # amber blinking
    FAULT = "FAULT"  # red, steady or blinking
    OFF = "OFF"  # nothing lit


@dataclass(frozen=True)
class StateEvent:
    machine: str
    from_state: State
    to_state: State
    ts: float  # when the new state began (back-dated to its first stable observation)
    duration_s: float  # how long the previous state lasted

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": self.machine,
            "from": self.from_state.value,
            "to": self.to_state.value,
            "ts": round(self.ts, 3),
            "duration_s": round(self.duration_s, 3),
        }


def classify(modes: dict[str, LampMode]) -> State:
    """Priority red > amber > green; nothing lit is OFF."""
    if modes.get("red", LampMode.OFF) != LampMode.OFF:
        return State.FAULT
    amber = modes.get("amber", LampMode.OFF)
    if amber == LampMode.BLINK:
        return State.SETUP
    if amber == LampMode.STEADY:
        return State.IDLE
    if modes.get("green", LampMode.OFF) != LampMode.OFF:
        return State.RUNNING
    return State.OFF


class MachineStateMachine:
    """Commits a state change only after the candidate has persisted for `min_dwell_s`."""

    def __init__(self, machine: str, min_dwell_s: float, start_ts: float = 0.0) -> None:
        self.machine, self.min_dwell_s = machine, min_dwell_s
        self.state: State | None = None
        self.initial_state: State | None = None
        self.started_at = start_ts
        self._pending: State | None = None
        self._pending_since = 0.0

    def update(self, ts: float, candidate: State, since: float | None = None) -> StateEvent | None:
        """Feed one observation. `since` may back-date when the candidate really began
        (e.g. the first edge of a blink run, which is only recognised a moment later)."""
        if self.state is None:  # first observation defines the initial state, no event
            self.state = self.initial_state = candidate
            return None
        if candidate == self.state:
            self._pending = None
            return None
        if candidate != self._pending:
            self._pending = candidate
            self._pending_since = max(self.started_at, min(ts, ts if since is None else since))
        if ts - self._pending_since + 1e-9 < self.min_dwell_s:
            return None
        event = StateEvent(self.machine, self.state, candidate, self._pending_since,
                           self._pending_since - self.started_at)
        self.state, self.started_at, self._pending = candidate, self._pending_since, None
        return event
