import numpy as np
import pytest

from stacklight.config import Thresholds
from stacklight.detect import LampDetector, rgb_to_hsv, scene_gain


@pytest.mark.parametrize(
    "rgb, expected",
    [
        ((255, 0, 0), (0.0, 1.0, 1.0)),
        ((0, 255, 0), (120.0, 1.0, 1.0)),
        ((0, 0, 255), (240.0, 1.0, 1.0)),
        ((255, 255, 0), (60.0, 1.0, 1.0)),
        ((0, 255, 255), (180.0, 1.0, 1.0)),
        ((255, 0, 255), (300.0, 1.0, 1.0)),
        ((255, 255, 255), (0.0, 0.0, 1.0)),
        ((0, 0, 0), (0.0, 0.0, 0.0)),
        ((128, 128, 128), (0.0, 0.0, 128 / 255)),
        ((255, 128, 0), (30.12, 1.0, 1.0)),
        ((255, 0, 128), (329.88, 1.0, 1.0)),
        ((100, 50, 50), (0.0, 0.5, 100 / 255)),
    ],
)
def test_rgb_to_hsv_known_values(rgb, expected):
    h, s, v = rgb_to_hsv(np.array(rgb, dtype=np.uint8))
    assert h == pytest.approx(expected[0], abs=0.05)
    assert s == pytest.approx(expected[1], abs=1e-3)
    assert v == pytest.approx(expected[2], abs=1e-3)


def test_rgb_to_hsv_is_vectorised_and_in_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    h, s, v = rgb_to_hsv(img)
    assert h.shape == s.shape == v.shape == (20, 30)
    assert h.min() >= 0 and h.max() < 360
    assert s.min() >= 0 and s.max() <= 1 and v.max() <= 1


LENS = 20  # ROI side in pixels


def make_frame(ambient: float, hue_rgb: tuple[float, float, float], lit: bool) -> np.ndarray:
    """60x60 frame: grey wall scaled by `ambient`, one lens in the top-left 20x20 corner."""
    img = np.full((60, 60, 3), 0.4 * ambient, dtype=np.float32)
    lens = np.array(hue_rgb, dtype=np.float32)
    img[:LENS, :LENS] = np.clip(lens * 0.30 * ambient, 0, 1)  # dull unlit lens, scales with light
    if lit:
        img[:LENS, :LENS] = np.maximum(img[:LENS, :LENS], lens * 0.95)  # emissive, light independent
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


@pytest.mark.parametrize("color, rgb", [("red", (1.0, 0.05, 0.05)), ("amber", (1.0, 0.6, 0.0)),
                                        ("green", (0.1, 0.9, 0.2))])
def test_detects_each_colour(color, rgb):
    det = LampDetector(color, (0, 0, LENS, LENS), Thresholds())
    assert not det.update(make_frame(1.0, rgb, False), 0.4).on
    assert det.update(make_frame(1.0, rgb, True), scene_gain(make_frame(1.0, rgb, True))).on


def test_wrong_colour_is_not_detected():
    det = LampDetector("green", (0, 0, LENS, LENS), Thresholds())
    frame = make_frame(1.0, (1.0, 0.05, 0.05), True)  # a lit RED lens
    assert not det.update(frame, scene_gain(frame)).on


def test_robust_to_brightness_drift_and_light_switch():
    """Ambient sweeps 1.2 -> 0.3, jumps to 1.2 and back; verdicts must stay correct."""
    rgb = (1.0, 0.05, 0.05)
    det = LampDetector("red", (0, 0, LENS, LENS), Thresholds())
    ambients = list(np.linspace(1.2, 0.3, 40)) + [1.2] * 10 + [0.3] * 10 + list(np.linspace(0.3, 1.2, 30))
    wrong = 0
    for i, amb in enumerate(ambients):
        lit = (i // 10) % 2 == 1  # toggle the lamp every 10 frames
        frame = make_frame(float(amb), rgb, lit)
        wrong += det.update(frame, scene_gain(frame)).on != lit
    assert wrong == 0


def test_baseline_not_learned_while_lamp_is_on():
    rgb = (0.1, 0.9, 0.2)
    det = LampDetector("green", (0, 0, LENS, LENS), Thresholds())
    off, on = make_frame(1.0, rgb, False), make_frame(1.0, rgb, True)
    det.update(off, scene_gain(off))
    baseline = det.baseline
    for _ in range(500):  # lamp stays on for a long time
        assert det.update(on, scene_gain(on)).on
    assert det.baseline == baseline
