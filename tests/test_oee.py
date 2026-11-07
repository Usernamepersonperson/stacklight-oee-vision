import pytest

from stacklight.oee import compute_oee, intervals_from_events
from stacklight.state import State, StateEvent

R, I, S, F, O = State.RUNNING, State.IDLE, State.SETUP, State.FAULT, State.OFF


def ev(ts: float, frm: State, to: State) -> StateEvent:
    return StateEvent("m1", frm, to, ts, 0.0)


# Hand-built 600 s timeline:
#   0-100 RUN | 100-120 FAULT | 120-300 RUN | 300-330 IDLE | 330-420 RUN
#   420-450 FAULT | 450-480 SETUP | 480-540 RUN | 540-600 OFF
EVENTS = [ev(100, R, F), ev(120, F, R), ev(300, R, I), ev(330, I, R),
          ev(420, R, F), ev(450, F, S), ev(480, S, R), ev(540, R, O)]


def test_intervals_cover_the_window_exactly():
    ivs = intervals_from_events(EVENTS, R, 0, 600)
    assert ivs[0].start == 0 and ivs[-1].end == 600
    assert all(a.end == b.start for a, b in zip(ivs, ivs[1:]))
    assert sum(i.length for i in ivs) == 600


def test_hand_computed_oee():
    rep = compute_oee("m1", intervals_from_events(EVENTS, R, 0, 600))
    # run = 100 + 180 + 90 + 60 = 430; fault = 20 + 30 = 50; idle 30; setup 30; off 60
    assert rep.run_s == 430
    assert rep.seconds_by_state == {"RUNNING": 430, "IDLE": 30, "SETUP": 30, "FAULT": 50, "OFF": 60}
    assert rep.planned_s == 540  # OFF excluded
    assert rep.availability == pytest.approx(430 / 540)
    assert rep.failures == 2
    assert rep.mtbf_s == pytest.approx(215)
    assert rep.mttr_s == pytest.approx(25)


def test_include_off_counts_off_as_planned_time():
    rep = compute_oee("m1", intervals_from_events(EVENTS, R, 0, 600), include_off=True)
    assert rep.availability == pytest.approx(430 / 600)


def test_no_faults_gives_no_mtbf_mttr():
    rep = compute_oee("m1", intervals_from_events([ev(50, R, I)], R, 0, 100))
    assert rep.availability == pytest.approx(0.5)
    assert rep.failures == 0 and rep.mtbf_s is None and rep.mttr_s is None


def test_all_off_has_no_availability():
    rep = compute_oee("m1", intervals_from_events([], O, 0, 100))
    assert rep.availability is None


def test_events_after_window_end_are_ignored():
    ivs = intervals_from_events([ev(50, R, F), ev(500, F, R)], R, 0, 100)
    rep = compute_oee("m1", ivs)
    assert rep.run_s == 50 and rep.seconds_by_state["FAULT"] == 50
