"""Planted scan defects are measured, and a clean page calls for no correction."""

from pathlib import Path

import numpy as np

from llb.prep.book.preprocess.constants import (
    BLUR_SHARP_MIN,
    BOX_TOLERANCE_PX,
    CONTRAST_CORRECT,
    CURVATURE_CORRECT_PX,
    GUTTER_WIDTH_CORRECT,
    NOISE_CORRECT,
    SKEW_CORRECT_DEG,
    UNEVENNESS_CORRECT,
)
from llb.prep.book.preprocess.decide import selected
from llb.prep.book.preprocess.images import write_png
from llb.prep.book.preprocess.measure import measure_page
from tests.llb.prep.book._page_factory import (
    add_noise,
    bleed_page,
    blur,
    bow,
    clean_page,
    darken_gutter,
    fade,
    tilt,
    uneven,
    upside_down,
)

_NAMES = ("rotated", "deskew", "dewarp", "illumination", "contrast", "gutter", "denoise")


def _signals(page: np.ndarray, name: str = "p0001"):
    return measure_page(page, name, 1)


def test_a_clean_page_calls_for_no_correction():
    signals = _signals(clean_page())
    assert signals.orientation_deg == 0
    assert abs(signals.skew_deg) < SKEW_CORRECT_DEG
    assert signals.line_curvature_px < CURVATURE_CORRECT_PX
    assert signals.background_unevenness < UNEVENNESS_CORRECT
    assert signals.contrast > CONTRAST_CORRECT
    assert signals.blur > BLUR_SHARP_MIN
    assert signals.noise < NOISE_CORRECT
    assert signals.gutter_side == "none"
    assert signals.bleed is None
    assert not any(selected(signals).values())


def test_each_planted_defect_is_detected():
    straight = clean_page()
    skewed = _signals(tilt(straight, 2.0))
    curved = _signals(bow(straight, 8.0))
    shaded = _signals(uneven(straight))
    faint = _signals(fade(straight))
    soft = _signals(blur(straight))
    grainy = _signals(add_noise(straight))
    guttered = _signals(darken_gutter(straight))
    bled = _signals(bleed_page())

    assert abs(skewed.skew_deg - (-2.0)) <= 0.3
    assert curved.line_curvature_px >= CURVATURE_CORRECT_PX
    assert shaded.background_unevenness >= UNEVENNESS_CORRECT
    assert faint.contrast <= CONTRAST_CORRECT
    assert soft.blur < BLUR_SHARP_MIN
    assert grainy.noise >= NOISE_CORRECT
    assert guttered.gutter_side == "left"
    assert guttered.gutter_width_px >= GUTTER_WIDTH_CORRECT
    assert bled.bleed is not None and bled.bleed.side == "left"
    assert selected(skewed)["deskew"]
    assert selected(curved)["dewarp"]
    assert selected(shaded)["illumination"]
    assert selected(faint)["contrast"]
    assert selected(grainy)["denoise"]
    assert selected(guttered)["gutter"]
    assert "blur" not in selected(soft)


def test_an_upside_down_page_is_detected(tmp_path: Path):
    signals = _signals(upside_down(clean_page()))
    assert signals.orientation_deg == 180
    assert selected(signals)["rotated"]
    assert signals.blur > BLUR_SHARP_MIN


def test_signals_round_trip_and_do_not_rewrite_the_page(tmp_path: Path):
    page = clean_page()
    path = tmp_path / "page.png"
    write_png(path, page)
    before = path.read_bytes()
    signals = _signals(page)
    from llb.prep.book.preprocess.records import read_signals, write_signals

    target = tmp_path / "signals.json"
    write_signals(target, signals)
    assert read_signals(target) == signals
    assert path.read_bytes() == before
    assert page.tobytes() == clean_page().tobytes()


def test_tolerance_constant_is_the_stated_pixel_budget():
    assert BOX_TOLERANCE_PX == 4
