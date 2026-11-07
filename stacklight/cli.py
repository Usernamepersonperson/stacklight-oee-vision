"""Command line: `python -m stacklight simulate|camera`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable

from .config import Config, load_config
from .oee import OeeReport
from .pipeline import LineMonitor, run_frames
from .sim import Simulator, default_config, default_scenario, evaluate
from .state import StateEvent


def _fmt_event(ev: StateEvent) -> str:
    return (f"[{ev.ts:7.1f}s] {ev.machine:<10} {ev.from_state.value:<7} -> {ev.to_state.value:<7}"
            f" (previous state lasted {ev.duration_s:.1f}s)")


def _fmt_mtx(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def format_oee_table(reports: list[OeeReport]) -> str:
    head = (f"{'machine':<10} {'avail':>7} {'run s':>7} {'idle s':>7} {'setup s':>8} {'fault s':>8}"
            f" {'off s':>6} {'faults':>6} {'MTBF s':>7} {'MTTR s':>7}")
    rows = [head, "-" * len(head)]
    for r in reports:
        s = r.seconds_by_state
        av = "n/a" if r.availability is None else f"{r.availability * 100:.1f}%"
        rows.append(f"{r.machine:<10} {av:>7} {s['RUNNING']:>7.0f} {s['IDLE']:>7.0f} {s['SETUP']:>8.0f}"
                    f" {s['FAULT']:>8.0f} {s['OFF']:>6.0f} {r.failures:>6} {_fmt_mtx(r.mtbf_s):>7}"
                    f" {_fmt_mtx(r.mttr_s):>7}")
    return "\n".join(rows)


def _sinks(cfg: Config, args: argparse.Namespace) -> tuple[Callable[[StateEvent], None], Callable[[], None]]:
    """Build the event callback (print + optional MQTT) and a closer."""
    publisher = None
    if getattr(args, "mqtt", None):
        from .mqtt_out import MqttPublisher

        publisher = MqttPublisher(args.mqtt, cfg.line, args.mqtt_port)

    def on_event(ev: StateEvent) -> None:
        print(_fmt_event(ev), flush=True)
        if publisher:
            publisher.publish_event(ev)

    return on_event, (publisher.close if publisher else lambda: None)


def _write_jsonl(path: str, cfg: Config, events: list[StateEvent], reports: list[OeeReport]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps({"type": "event", "line": cfg.line, **ev.to_dict()}) + "\n")
        for r in reports:
            fh.write(json.dumps({"type": "oee", "line": cfg.line, **r.to_dict()}) + "\n")


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else default_config()
    scenario = default_scenario(cfg, args.duration, args.seed)
    monitor = LineMonitor(cfg)
    on_event, close = _sinks(cfg, args)
    print(f"Simulating {len(cfg.machines)} machines for {args.duration:.0f}s at {cfg.fps:g} fps"
          f" (noise, flicker, day/night drift, light switch)\n")
    try:
        run_frames(monitor, Simulator(cfg, scenario).frames(), on_event)
    finally:
        close()
    reports = monitor.oee(args.duration)
    print("\n" + format_oee_table(reports))
    ev = evaluate(scenario, monitor.events, monitor.initial_states)
    print(f"\nvs scripted ground truth: {ev.frame_accuracy * 100:.2f}% of frames labelled correctly, "
          f"{ev.matched}/{ev.transitions} transitions within 1s "
          f"(mean error {ev.mean_time_error_s:.2f}s), {ev.spurious} spurious events")
    print("Note: availability only. Performance and quality need part counts, which lamps do not carry.")
    if args.json:
        _write_jsonl(args.json, cfg, monitor.events, reports)
    return 0


def cmd_camera(args: argparse.Namespace) -> int:
    from .camera import iter_frames

    if not args.config:
        print("camera mode needs --config with ROIs in your camera's pixel coordinates", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    monitor = LineMonitor(cfg)
    on_event, close = _sinks(cfg, args)
    last = 0.0
    try:
        for ts, frame in iter_frames(args.source, cfg.fps):
            last = ts
            for ev in monitor.process(frame, ts):
                on_event(ev)
    except KeyboardInterrupt:
        pass
    finally:
        close()
    reports = monitor.oee(last)
    print("\n" + format_oee_table(reports))
    if args.json:
        _write_jsonl(args.json, cfg, monitor.events, reports)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stacklight", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", help="JSON config (default for simulate: built-in 3-machine line)")
        sp.add_argument("--json", metavar="OUT.jsonl", help="write events and OEE summary as JSON lines")
        sp.add_argument("--mqtt", metavar="HOST", help="publish to factory/<line>/<machine>/state")
        sp.add_argument("--mqtt-port", type=int, default=1883)

    s = sub.add_parser("simulate", help="synthetic video -> pipeline -> events + OEE")
    common(s)
    s.add_argument("--duration", type=float, default=300.0, help="seconds of video (default 300)")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_simulate)

    c = sub.add_parser("camera", help="live camera index or RTSP URL (needs opencv)")
    common(c)
    c.add_argument("--source", required=True, help="0 for a local camera, or rtsp://...")
    c.set_defaults(func=cmd_camera)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
