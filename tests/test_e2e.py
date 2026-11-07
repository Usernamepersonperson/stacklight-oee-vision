"""End to end: synthetic video with a scripted timeline -> pipeline -> compare with the script."""

import json
import sys

import pytest

from stacklight.cli import main
from stacklight.oee import compute_oee, intervals_from_events
from stacklight.pipeline import LineMonitor, run_frames
from stacklight.sim import Simulator, default_config, default_scenario, evaluate
from stacklight.state import State

DURATION = 180.0


@pytest.fixture(scope="module")
def result():
    cfg = default_config()
    scenario = default_scenario(cfg, DURATION, seed=7)
    monitor = LineMonitor(cfg)
    run_frames(monitor, Simulator(cfg, scenario).frames())
    return cfg, scenario, monitor, evaluate(scenario, monitor.events, monitor.initial_states, tol_s=1.0)


def test_cv2_is_not_needed():
    assert "cv2" not in sys.modules


def test_frame_accuracy_and_transition_timing(result, capsys):
    _, _, _, ev = result
    with capsys.disabled():
        print(f"\n[e2e] frames={ev.frames} accuracy={ev.frame_accuracy:.4f} "
              f"transitions={ev.matched}/{ev.transitions} mean_err={ev.mean_time_error_s:.2f}s "
              f"max_err={ev.max_time_error_s:.2f}s spurious={ev.spurious}")
    assert ev.frame_accuracy >= 0.95
    assert ev.matched == ev.transitions  # every scripted transition found within 1 s
    assert ev.max_time_error_s <= 1.0
    assert ev.spurious == 0


def test_scripts_cover_every_state_including_blink_variants(result):
    _, scenario, _, _ = result
    seen = {st for sc in scenario.scripts for st, _ in sc.timeline}
    assert seen == set(State)
    assert any(sc.fault_blinks for sc in scenario.scripts)


def test_measured_availability_close_to_scripted(result):
    cfg, scenario, monitor, _ = result
    for rep, script in zip(monitor.oee(DURATION), scenario.scripts):
        run = sum(s for st, s in script.timeline if st is State.RUNNING)
        off = sum(s for st, s in script.timeline if st is State.OFF)
        expected = run / (DURATION - off)
        assert rep.availability == pytest.approx(expected, abs=0.02)


def test_oee_helper_matches_monitor(result):
    _, _, monitor, _ = result
    name = monitor.cfg.machines[0].name
    evs = [e for e in monitor.events if e.machine == name]
    ivs = intervals_from_events(evs, monitor.initial_states[name], 0.0, DURATION)
    assert compute_oee(name, ivs).availability == pytest.approx(monitor.oee(DURATION)[0].availability)


def test_cli_simulate_writes_jsonl(tmp_path, capsys):
    out = tmp_path / "out.jsonl"
    assert main(["simulate", "--duration", "40", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "avail" in text and "ground truth" in text
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["type"] for r in rows} == {"event", "oee"}
    assert sum(r["type"] == "oee" for r in rows) == 3
