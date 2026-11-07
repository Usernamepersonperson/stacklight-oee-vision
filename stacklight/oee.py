"""Availability, MTBF, MTTR and downtime by reason from state events.

Only AVAILABILITY is measurable from lamps. OEE = availability x performance x quality;
performance needs an ideal cycle time and a part count, quality needs a good/total count.
A stack light carries neither, so this module does not invent them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .state import State, StateEvent


@dataclass(frozen=True)
class Interval:
    state: State
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class OeeReport:
    machine: str
    window_s: float
    planned_s: float
    run_s: float
    availability: float | None  # run / planned; None when nothing was planned
    seconds_by_state: dict[str, float]
    failures: int  # FAULT intervals
    mtbf_s: float | None  # run time per failure
    mttr_s: float | None  # mean FAULT duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": self.machine,
            "window_s": round(self.window_s, 3),
            "planned_s": round(self.planned_s, 3),
            "run_s": round(self.run_s, 3),
            "availability": None if self.availability is None else round(self.availability, 4),
            "seconds_by_state": {k: round(v, 3) for k, v in self.seconds_by_state.items()},
            "failures": self.failures,
            "mtbf_s": None if self.mtbf_s is None else round(self.mtbf_s, 3),
            "mttr_s": None if self.mttr_s is None else round(self.mttr_s, 3),
        }


def intervals_from_events(events: Iterable[StateEvent], initial: State,
                          t_start: float, t_end: float) -> list[Interval]:
    """Turn a machine's ordered events into back-to-back intervals covering [t_start, t_end]."""
    out: list[Interval] = []
    state, cursor = initial, t_start
    for ev in events:
        if ev.ts >= t_end:
            break
        if ev.ts > cursor:
            out.append(Interval(state, cursor, ev.ts))
            cursor = ev.ts
        state = ev.to_state
    if cursor < t_end:
        out.append(Interval(state, cursor, t_end))
    return out


def compute_oee(machine: str, intervals: list[Interval], include_off: bool = False) -> OeeReport:
    """Availability over the intervals. OFF time is excluded from planned time unless include_off."""
    secs = {s.value: 0.0 for s in State}
    failures = 0
    for iv in intervals:
        secs[iv.state.value] += iv.length
        failures += int(iv.state is State.FAULT and iv.length > 0)
    window = sum(secs.values())
    planned = window if include_off else window - secs[State.OFF.value]
    run = secs[State.RUNNING.value]
    return OeeReport(
        machine=machine, window_s=window, planned_s=planned, run_s=run,
        availability=run / planned if planned > 0 else None,
        seconds_by_state=secs, failures=failures,
        mtbf_s=run / failures if failures else None,
        mttr_s=secs[State.FAULT.value] / failures if failures else None,
    )
