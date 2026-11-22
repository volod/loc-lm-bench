"""The signal vector of one rendered page.

Orientation is the clockwise turn of the printed content, not the PDF placement angle. The
placement only decides whether a weak baseline vote may turn the page over: that vote is trusted
on a scan that was placed sideways, and ignored on a portrait placement, where the same vote also
fires on an upright table. Skew and curvature are measured after the turn is undone in memory.
"""

import cv2
import numpy as np

from llb.core.contracts.data_prep.book_preprocess import PageSignals
from llb.prep.book.preprocess.appearance import (
    bleed_band,
    blur_score,
    contrast,
    gutter,
    noise_score,
    unevenness,
)
from llb.prep.book.preprocess.constants import (
    BAND_FRACTION,
    BASELINE_LINE_GLYPHS,
    BASELINE_MIN_LINES,
    MAX_SKEW_DEG,
    MIN_LINE_SPAN,
    SCHEMA_VERSION,
    SIGNAL_SCHEMA_ID,
    SKEW_COARSE_STEP,
    SKEW_FINE_STEP,
)
from llb.prep.book.preprocess.geometry import rotate_upright
from llb.prep.book.preprocess.images import ink_mask, text_bands
from llb.prep.book.preprocess.orientation import orientation_deg


def measure_page(
    grey: np.ndarray, page_id: str, pdf_page: int, *, placed_sideways: bool = False
) -> PageSignals:
    """Every signal of `grey`, in the original render's pixel frame."""
    height, width = grey.shape[:2]
    orientation = orientation_deg(grey, placed_sideways=placed_sideways)
    upright, _ = rotate_upright(grey, orientation)
    side, width_px = gutter(grey)
    return PageSignals(
        schema_id=SIGNAL_SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        kind="book-ocr-page-signals",
        page_id=page_id,
        pdf_page=pdf_page,
        width=width,
        height=height,
        skew_deg=measure_skew(upright),
        line_curvature_px=measure_curvature(upright),
        background_unevenness=unevenness(grey),
        contrast=contrast(grey),
        blur=blur_score(grey),
        noise=noise_score(grey),
        gutter_side=side,  # type: ignore[arg-type]
        gutter_width_px=width_px,
        bleed=bleed_band(grey),
        orientation_deg=orientation,  # type: ignore[arg-type]
    )


def measure_skew(grey: np.ndarray) -> float:
    """The counter-clockwise rotation, in degrees, that levels the text lines.

    Letter baselines decide when the page has them. A projection-profile search is kept for a page
    of full-width lines, which is the synthetic case: on a real page that search locks onto a
    figure or a column edge and reports several degrees of tilt that the baselines do not have.
    """
    angle = _baseline_angle(grey)
    if angle is not None:
        return round(angle, 2)
    ink = ink_mask(grey).astype(np.float32)
    small = cv2.resize(ink, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    coarse = _best_angle(small, _steps(-MAX_SKEW_DEG, MAX_SKEW_DEG, SKEW_COARSE_STEP))
    fine_lo = coarse - SKEW_COARSE_STEP
    fine_hi = coarse + SKEW_COARSE_STEP
    return round(_best_angle(small, _steps(fine_lo, fine_hi, SKEW_FINE_STEP)), 2)


def measure_curvature(grey: np.ndarray) -> float:
    """The median line sagitta, in pixels. A straight page is near zero."""
    ink = ink_mask(grey)
    values = [_sagitta(ink, y0, y1) for y0, y1 in text_bands(ink, BAND_FRACTION)]
    if not values:
        return 0.0
    return round(float(np.median(values)), 2)


def _baseline_angle(grey: np.ndarray) -> float | None:
    """Median tilt of the letter lines, or None when there are not enough long lines."""
    from llb.prep.book.preprocess.orientation import _letters, _lines

    angles = [angle for line in _lines(_letters(grey)) if (angle := _line_angle(line)) is not None]
    if len(angles) < BASELINE_MIN_LINES:
        return None
    return float(np.median(np.asarray(angles, dtype=np.float64)))


def _line_angle(line: list[tuple[int, int, int, int]]) -> float | None:
    if len(line) < BASELINE_LINE_GLYPHS:
        return None
    xs = np.asarray([box[0] + box[2] / 2.0 for box in line], dtype=np.float64)
    ys = np.asarray([box[1] + box[3] / 2.0 for box in line], dtype=np.float64)
    span = float(xs.max() - xs.min())
    if span < MIN_LINE_SPAN:
        return None
    order = np.argsort(xs)
    ys = ys[order]
    third = max(len(ys) // 3, 2)
    rise = float(np.median(ys[-third:]) - np.median(ys[:third]))
    return float(np.degrees(np.arctan(rise / span)))


def _best_angle(ink: np.ndarray, angles: list[float]) -> float:
    best_angle, best_score = 0.0, -1.0
    for angle in angles:
        score = _profile_var(ink, angle)
        if score > best_score:
            best_angle, best_score = angle, score
    return best_angle


def _profile_var(ink: np.ndarray, angle: float) -> float:
    height, width = ink.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
    return float(np.var(np.asarray(rotated).sum(axis=1)))


def _steps(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + index * step, 4) for index in range(count + 1)]


def _sagitta(ink: np.ndarray, y0: int, y1: int) -> float:
    top = max(0, y0 - 6)
    bottom = min(ink.shape[0], y1 + 6)
    band = ink[top:bottom]
    xs, ys = _centroids(band, top)
    if len(xs) < 8:
        return 0.0
    axis = np.asarray(xs, dtype=np.float64)
    linear = np.polyval(np.polyfit(xs, ys, 1), axis)
    curved = np.polyval(np.polyfit(xs, ys, 2), axis)
    return float(np.max(np.abs(curved - linear)))


def _centroids(band: np.ndarray, top: int) -> tuple[list[int], list[float]]:
    height, width = band.shape
    step = max(1, width // 40)
    xs: list[int] = []
    ys: list[float] = []
    rows = np.arange(height, dtype=np.float64)
    for x in range(0, width, step):
        column = band[:, x].astype(np.float64)
        total = float(column.sum())
        if total < 1:
            continue
        xs.append(x)
        ys.append(top + float(np.dot(rows, column) / total))
    return xs, ys
