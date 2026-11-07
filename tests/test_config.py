from pathlib import Path

import pytest

from stacklight.config import config_from_dict, load_config
from stacklight.sim import default_config

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "line_config.json"


def test_example_config_loads_and_matches_simulator_layout():
    cfg = load_config(EXAMPLE)
    assert len(cfg.machines) == 3
    assert cfg == default_config()  # the example is what the simulator draws


def minimal(**over):
    d = {"machines": [{"name": "a", "lamps": [{"color": "red", "roi": [0, 0, 5, 5]}]}]}
    d.update(over)
    return d


def test_minimal_config_gets_defaults():
    cfg = config_from_dict(minimal())
    assert cfg.fps == 10 and cfg.line == "line-1"


@pytest.mark.parametrize(
    "bad",
    [
        {"machines": []},
        minimal(fps=0),
        minimal(thresholds={"nope": 1}),
        {"machines": [{"name": "a", "lamps": [{"color": "blue", "roi": [0, 0, 5, 5]}]}]},
        {"machines": [{"name": "a", "lamps": [{"color": "red", "roi": [0, 0, 0, 5]}]}]},
        {"machines": [{"name": "a", "lamps": [{"color": "red", "roi": [0, 0, 5, 5]},
                                              {"color": "red", "roi": [9, 0, 5, 5]}]}]},
    ],
)
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        config_from_dict(bad)
