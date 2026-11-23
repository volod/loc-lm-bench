"""Which corrections a signal vector calls for. The thresholds live in one place."""

from llb.core.contracts.data_prep.book_preprocess import PageSignals
from llb.prep.book.preprocess.constants import (
    CONTRAST,
    CONTRAST_CORRECT,
    CURVATURE_CORRECT_PX,
    DENOISE,
    DEWARP,
    DESKEW,
    GUTTER,
    GUTTER_MAX_FRACTION,
    GUTTER_WIDTH_CORRECT,
    ILLUMINATION,
    NOISE_CORRECT,
    ROTATED,
    SKEW_CORRECT_DEG,
    UNEVENNESS_CORRECT,
)


def selected(signals: PageSignals) -> dict[str, bool]:
    """The correction name to whether this page's signals call for it."""
    return {
        ROTATED: signals.orientation_deg != 0,
        DESKEW: abs(signals.skew_deg) >= SKEW_CORRECT_DEG,
        DEWARP: signals.line_curvature_px >= CURVATURE_CORRECT_PX,
        ILLUMINATION: signals.background_unevenness >= UNEVENNESS_CORRECT,
        CONTRAST: signals.contrast <= CONTRAST_CORRECT,
        GUTTER: _gutter(signals),
        DENOISE: signals.noise >= NOISE_CORRECT,
    }


def _gutter(signals: PageSignals) -> bool:
    """A narrow dark edge is a binding shadow. A dark region across the page is a graphic."""
    if signals.gutter_side == "none" or signals.gutter_width_px < GUTTER_WIDTH_CORRECT:
        return False
    span = signals.width if signals.gutter_side in ("left", "right") else signals.height
    return signals.gutter_width_px <= span * GUTTER_MAX_FRACTION
