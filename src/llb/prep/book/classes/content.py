"""Cheap content signals of one rendered page.

Formula density, ruling lines, figure area, column count, and the footnote band are measured from
the ink of the page. Nothing here reads a character. A page larger than the analysis size is
reduced first; every threshold is a fraction of that image or of its own letter height, so a
fixture and a full-resolution scan use the same rules.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from llb.prep.book.classes.constants import ANALYSIS_LONG_SIDE, FORMULA_MIN_LINES
from llb.prep.book.preprocess.geometry import rotate_upright

_GLYPH_AREA = (4, 500)
_GLYPH_HEIGHT = 48


@dataclass(frozen=True)
class ContentSignals:
    """Scale-free content measurements. Fractions are in 0..1."""

    formula_density: float
    equation_lines: int
    ruling_lines: int
    vertical_rules: int
    ruled_fraction: float
    figure_area: float
    diagram_labels: int
    column_count: int
    footnote_band: float
    corner_components: int
    gutter_ink: float


def measure_content(grey: np.ndarray, orientation_deg: int = 0) -> ContentSignals:
    """Content signals of `grey`, after turning a sideways or upside-down page upright."""
    from llb.prep.book.classes.layout import (
        column_spans,
        corner_count,
        drawings,
        margin_ink,
        rule_grid,
    )
    from llb.prep.book.classes.textsig import equation_hits, footnote_band

    upright, _ = rotate_upright(np.asarray(grey, dtype=np.uint8), int(orientation_deg))
    image = resize_analysis(upright)
    ink = ink_mask(image)
    count, stats, _centroids = components(ink)
    x_height = letter_height(stats, count)
    columns, spans = column_spans(ink)
    mask = light_mask(image)
    hits, lines = equation_hits(ink, mask, x_height, spans)
    if lines < FORMULA_MIN_LINES:
        hits, density = 0, 0.0
    else:
        density = unit(hits / lines) if lines else 0.0
    run, vertical, span = rule_grid(mask, x_height)
    area, labels = drawings(stats, count, ink, x_height)
    return ContentSignals(
        formula_density=density,
        equation_lines=hits,
        ruling_lines=run,
        vertical_rules=vertical,
        ruled_fraction=unit(span),
        figure_area=unit(area),
        diagram_labels=labels,
        column_count=columns,
        footnote_band=unit(footnote_band(mask, ink, x_height)),
        corner_components=corner_count(stats, count, ink.shape, x_height),
        gutter_ink=unit(margin_ink(ink)),
    )


def resize_analysis(grey: np.ndarray) -> np.ndarray:
    """Shrink the long side to the analysis size. A smaller page is unchanged."""
    height, width = grey.shape[:2]
    long_side = max(height, width)
    if long_side <= ANALYSIS_LONG_SIDE:
        return grey
    scale = ANALYSIS_LONG_SIDE / float(long_side)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.asarray(cv2.resize(grey, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def ink_mask(grey: np.ndarray) -> np.ndarray:
    """Pixels darker than three quarters of a wide local background."""
    limit = int(min(grey.shape[:2]))
    kernel = min(61, max(3, limit // 12))
    if kernel % 2 == 0:
        kernel += 1
    kernel = min(kernel, limit if limit % 2 == 1 else limit - 1)
    background = cv2.medianBlur(grey, max(kernel, 3))
    return grey.astype(np.int16) * 4 < background.astype(np.int16) * 3


def light_mask(image: np.ndarray) -> np.ndarray:
    """Pixels a little darker than a local median. Rules fainter than the ink mask still show."""
    limit = int(min(image.shape[:2]))
    kernel = min(31, max(3, limit // 15))
    if kernel % 2 == 0:
        kernel += 1
    background = cv2.medianBlur(image, kernel)
    return image.astype(np.int16) + 10 < background.astype(np.int16)


def components(ink: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Connected components of an ink mask: count, stats, centroids."""
    labelled = ink.astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(labelled, connectivity=8)
    return int(count), stats, centroids


def letter_height(stats: np.ndarray, count: int) -> float:
    """70th percentile height of glyph-sized components. A page of rules falls back to 8 px."""
    heights = []
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if _GLYPH_AREA[0] <= area <= _GLYPH_AREA[1] and height <= _GLYPH_HEIGHT:
            if width <= max(8, height * 3):
                heights.append(height)
    if len(heights) < 8:
        return 8.0
    return float(np.percentile(np.asarray(heights, dtype=np.float64), 70))


def bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-exclusive runs of True values along one axis."""
    found: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(np.asarray(mask).tolist()):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            found.append((start, index))
            start = None
    if start is not None:
        found.append((start, len(mask)))
    return found


def unit(value: float) -> float:
    """A fraction clamped to 0..1 and rounded so two runs write the same digits."""
    return round(min(1.0, max(0.0, float(value))), 4)
