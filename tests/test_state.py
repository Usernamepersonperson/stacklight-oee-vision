import pytest

from stacklight.state import MachineStateMachine, State, classify
from stacklight.temporal import LampMode as M


@pytest.mark.parametrize(
    "modes, expected",
    [
        ({}, State.OFF),
        ({"green": M.STEADY}, State.RUNNING),
        ({"amber": M.STEADY}, State.IDLE),
        ({"amber": M.BLINK}, State.SETUP),
        ({"red": M.STEADY}, State.FAULT),
        ({"red": M.BLINK}, State.FAULT),
        ({"red": M.STEADY, "green": M.STEADY}, State.FAULT),  # red wins
        ({"amber": M.STEADY, "green": M.STEADY}, State.IDLE),  # amber over green
        ({"red": M.OFF, "amber": M.OFF, "green": M.OFF}, State.OFF),
    ],
)
def test_classify(modes, expected):
    assert classify(modes) is expected


def drive(sm: MachineStateMachine, seq: list[tuple[float, State]]):
    return [ev for ts, st in seq if (ev := sm.update(ts, st))]


def test_first_observation_sets_initial_state_without_event():
    sm = MachineStateMachine("m1", 1.0)
    assert sm.update(0.0, State.RUNNING) is None
    assert sm.state is State.RUNNING and sm.initial_state is State.RUNNING


def test_short_blip_is_filtered_by_dwell():
    sm = MachineStateMachine("m1", 1.0)
    seq = [(i / 10, State.RUNNING) for i in range(30)]
    seq += [(3.0 + i / 10, State.FAULT) for i in range(5)]  # 0.5 s blip
    seq += [(3.5 + i / 10, State.RUNNING) for i in range(30)]
    assert drive(sm, seq) == []
    assert sm.state is State.RUNNING


def test_change_is_committed_after_dwell_and_back_dated():
    sm = MachineStateMachine("m1", 1.0)
    seq = [(i / 10, State.RUNNING) for i in range(50)]  # 0.0 .. 4.9
    seq += [(5.0 + i / 10, State.IDLE) for i in range(30)]  # from 5.0
    (ev,) = drive(sm, seq)
    assert (ev.from_state, ev.to_state) == (State.RUNNING, State.IDLE)
    assert ev.ts == pytest.approx(5.0)
    assert ev.duration_s == pytest.approx(5.0)
    assert ev.to_dict() == {"machine": "m1", "from": "RUNNING", "to": "IDLE", "ts": 5.0, "duration_s": 5.0}


def test_flapping_candidates_restart_the_dwell_timer():
    sm = MachineStateMachine("m1", 1.0)
    sm.update(0.0, State.RUNNING)
    seq = []
    for i in range(1, 30):  # alternate IDLE/FAULT every 0.5 s: neither lasts 1 s
        seq.append((i * 0.1, State.IDLE if (i // 5) % 2 == 0 else State.FAULT))
    assert drive(sm, seq) == []


def test_since_hint_back_dates_the_event():
    sm = MachineStateMachine("m1", 1.0)
    sm.update(0.0, State.RUNNING)
    ev = sm.update(5.0, State.SETUP, since=3.9)
    assert ev is not None and ev.ts == pytest.approx(3.9)
