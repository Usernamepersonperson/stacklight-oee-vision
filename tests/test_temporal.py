import pytest

from stacklight.config import Thresholds
from stacklight.temporal import LampMode, LampTracker

FPS = 10.0


def run(pattern: list[bool]) -> list[LampMode]:
    tr = LampTracker(FPS, Thresholds())
    return [tr.update(x) for x in pattern]


def blink(hz: float, seconds: float, phase: float = 0.0) -> list[bool]:
    return [((i / FPS) * hz + phase) % 1.0 < 0.5 for i in range(int(seconds * FPS))]


def test_steady_on_and_off():
    assert run([True] * 30)[-1] is LampMode.STEADY
    assert run([False] * 30)[-1] is LampMode.OFF


@pytest.mark.parametrize("hz", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("phase", [0.0, 0.3, 0.7])
def test_blink_is_detected_and_distinct_from_steady(hz, phase):
    modes = run(blink(hz, 6.0, phase))
    assert modes[-1] is LampMode.BLINK
    assert all(m is LampMode.BLINK for m in modes[30:])  # stays BLINK, never drops to steady/off


def test_single_frame_flicker_does_not_look_like_a_blink():
    pattern = [True] * 60
    for i in (15, 31, 47):
        pattern[i] = False
    modes = run(pattern)
    assert set(modes[5:]) == {LampMode.STEADY}


def test_slow_toggle_is_not_a_blink():
    pattern = ([True] * 20 + [False] * 20) * 2  # 0.25 Hz
    assert LampMode.BLINK not in run(pattern)


def test_blink_to_steady_and_back_within_a_second():
    pattern = blink(1.5, 4.0) + [True] * 15 + blink(1.5, 3.0, 0.5)
    modes = run(pattern)
    assert modes[39] is LampMode.BLINK
    assert modes[40 + 8] is LampMode.STEADY  # < 1 s after the blink ends
    assert modes[-1] is LampMode.BLINK


def test_blink_onset_age_points_at_first_edge():
    tr = LampTracker(FPS, Thresholds())
    for x in [True] * 20 + blink(1.5, 1.0, 0.5):
        tr.update(x)
    assert tr.mode is LampMode.BLINK
    assert 0.5 <= tr.blink_age_s <= 1.4
