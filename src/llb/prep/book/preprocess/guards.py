"""Refuse a variant that erases a thin stroke or drops a line off the page.

The stroke guard looks at minus signs, decimal points, and rules measured on the original. The
content guard looks at text lines and at ink in the outer band of the page. A photometric variant
is judged only on strokes, because it does not move the page. A geometric variant is also judged
on coverage: a point map that does not reach the page is a crop, whatever the pixels look like.
"""

import cv2
import numpy as np

from llb.prep.book.preprocess.constants import (
    BAND_FRACTION,
    CONTENT_WINDOW,
    DOT_MAX_AREA,
    DOT_MAX_SIDE,
    DOT_MIN_AREA,
    EDGE_FRACTION,
    LINE_MAX_HEIGHT,
    STROKE_DARK_MAX,
    STROKE_INK_MAX,
    STROKE_KEEP_FRACTION,
    STROKE_MAX_HEIGHT,
    STROKE_MIN_WIDTH,
    STROKE_WINDOW,
)
from llb.prep.book.preprocess.images import ink_mask, text_bands
from llb.prep.book.preprocess.maps import Locate

Box = tuple[int, int, int, int]


def refusal(
    original: np.ndarray,
    variant: np.ndarray,
    locate: Locate,
    *,
    geometric: bool,
    covered: bool,
) -> str | None:
    """Why `variant` is refused, or None when it may be kept."""
    if geometric and not covered:
        return "point map does not cover the page"
    if geometric:
        lost = _lost_lines(original, variant, locate)
        if lost:
            return lost
    return _lost_strokes(original, variant, locate)


def _lost_lines(original: np.ndarray, variant: np.ndarray, locate: Locate) -> str | None:
    points = _line_points(original)
    points.extend(_edge_ink(original))
    if not points:
        return None
    lost = sum(1 for point in points if not _dark_at(variant, locate, point, CONTENT_WINDOW))
    if lost:
        return f"lost {lost} text line(s) or page-edge line(s)"
    return None


def _lost_strokes(original: np.ndarray, variant: np.ndarray, locate: Locate) -> str | None:
    strokes = thin_strokes(original)
    if not strokes:
        return None
    lost = sum(1 for box in strokes if not _dark_at(variant, locate, _center(box), STROKE_WINDOW))
    if lost / len(strokes) > 1.0 - STROKE_KEEP_FRACTION:
        return f"erased {lost} of {len(strokes)} thin strokes"
    return None


def thin_strokes(grey: np.ndarray) -> list[Box]:
    """Minus signs, rules, and decimal-point-sized marks on `grey`."""
    ink = ink_mask(grey).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    found = [_component(grey, stats[index]) for index in range(1, count)]
    return _spread([box for box in found if box is not None], 80)


def _component(grey: np.ndarray, stats: np.ndarray) -> Box | None:
    x, y, width, height, area = (int(stats[i]) for i in range(5))
    if int(np.median(grey[y : y + height, x : x + width])) > STROKE_INK_MAX:
        return None
    thin = height <= STROKE_MAX_HEIGHT and width >= STROKE_MIN_WIDTH
    dot = (
        DOT_MIN_AREA <= area <= DOT_MAX_AREA
        and max(width, height) <= DOT_MAX_SIDE
        and _on_paper(grey, x, y, width, height)
    )
    if thin or dot:
        return (x, y, x + width, y + height)
    return None


def _on_paper(grey: np.ndarray, x: int, y: int, width: int, height: int) -> bool:
    """A decimal point sits on blank paper. A speck inside a photograph does not."""
    y0, y1 = max(0, y - 3), min(grey.shape[0], y + height + 3)
    x0, x1 = max(0, x - 3), min(grey.shape[1], x + width + 3)
    return float(np.median(grey[y0:y1, x0:x1])) >= STROKE_DARK_MAX


def _line_points(grey: np.ndarray) -> list[tuple[float, float]]:
    """One ink pixel per text line. The middle of the page may be a column gap."""
    ink = ink_mask(grey)
    points: list[tuple[float, float]] = []
    for y0, y1 in text_bands(ink, BAND_FRACTION):
        if y1 - y0 > LINE_MAX_HEIGHT:
            continue
        point = _band_point(ink, y0, y1)
        if point is not None:
            points.append(point)
    return points


def _band_point(ink: np.ndarray, y0: int, y1: int) -> tuple[float, float] | None:
    columns = np.flatnonzero(ink[y0:y1].any(axis=0))
    if columns.size == 0:
        return None
    x = int(columns[int(columns.size // 2)])
    rows = np.flatnonzero(ink[y0:y1, x])
    if rows.size == 0:
        return None
    return float(x), float(y0 + int(rows[int(rows.size // 2)]))


def _edge_ink(grey: np.ndarray) -> list[tuple[float, float]]:
    """One ink pixel in each outer strip that actually contains ink."""
    ink = ink_mask(grey)
    height, width = ink.shape
    edge = max(2, int(min(height, width) * EDGE_FRACTION))
    regions = (
        (0, 0, width, edge),
        (0, height - edge, width, height),
        (0, 0, edge, height),
        (width - edge, 0, width, height),
    )
    found = (_region_point(ink, *region) for region in regions)
    return [point for point in found if point is not None]


def _dark_at(image: np.ndarray, locate: Locate, point: tuple[float, float], radius: int) -> bool:
    found = locate(point[0], point[1])
    if found is None:
        return False
    x, y = int(round(found[0])), int(round(found[1]))
    height, width = image.shape[:2]
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return False
    return bool(int(image[y0:y1, x0:x1].min()) <= STROKE_DARK_MAX)


def _region_point(
    ink: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> tuple[float, float] | None:
    ys, xs = np.nonzero(ink[y0:y1, x0:x1])
    if xs.size == 0:
        return None
    mid = int(xs.size // 2)
    return float(x0 + int(xs[mid])), float(y0 + int(ys[mid]))


def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _spread(boxes: list[Box], limit: int) -> list[Box]:
    if len(boxes) <= limit:
        return boxes
    step = len(boxes) / limit
    return [boxes[int(index * step)] for index in range(limit)]
