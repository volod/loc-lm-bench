"""Flatten bowed text lines with a vertical remap whose point map we record.

The shift is the bow only: the quadratic fit of a text line minus its straight fit. It is zero
outside that line's horizontal span, so the fit is not extrapolated into the margin, and it fades
to zero above the first line and below the last, so a rule on the page edge stays put. A line
bowed past the cap, or several times the page's typical line, is left out -- that fit is a figure
or a column edge, not curl. A remap that would send the page outside itself is the caller's to
refuse.
"""

import cv2
import numpy as np

from llb.prep.book.preprocess.constants import (
    BAND_FRACTION,
    BOW_OUTLIER_RATIO,
    CURVATURE_CORRECT_PX,
    MAX_LINE_BOW_PX,
)
from llb.prep.book.preprocess.images import ink_mask, text_bands


def unbend(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the flattened page and the variant-to-original `map_x`, `map_y`."""
    height, width = grey.shape[:2]
    shift = _shift_field(ink_mask(grey), height, width)
    ys, xs = np.mgrid[0:height, 0:width]
    map_x = xs.astype(np.float32)
    map_y = (ys.astype(np.float32) + shift).astype(np.float32)
    image = cv2.remap(grey, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.asarray(image, dtype=np.uint8), map_x, map_y


def _shift_field(ink: np.ndarray, height: int, width: int) -> np.ndarray:
    found: list[tuple[float, np.ndarray]] = []
    for y0, y1 in text_bands(ink, BAND_FRACTION):
        sample = _line_shift(ink, y0, y1, width)
        if sample is not None:
            found.append(sample)
    found = _text_bows(found)
    if not found:
        return np.zeros((height, width), dtype=np.float32)
    centres = np.asarray([sample[0] for sample in found], dtype=np.float64)
    curves = np.vstack([sample[1] for sample in found])
    order = np.argsort(centres)
    centres, curves = centres[order], curves[order]
    return _interpolate(centres, curves, height, width)


def _line_shift(ink: np.ndarray, y0: int, y1: int, width: int) -> tuple[float, np.ndarray] | None:
    top = max(0, y0 - 6)
    bottom = min(ink.shape[0], y1 + 6)
    band = ink[top:bottom]
    xs, ys = _centroids(band, top)
    if len(xs) < 8:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    order = np.argsort(x_arr)
    x_arr, y_arr = x_arr[order], y_arr[order]
    if float(x_arr[-1] - x_arr[0]) < 1.0:
        return None
    bow = np.polyval(np.polyfit(x_arr, y_arr, 2), x_arr) - np.polyval(
        np.polyfit(x_arr, y_arr, 1), x_arr
    )
    if float(np.max(np.abs(bow))) > MAX_LINE_BOW_PX:
        return None
    # Zero outside the samples. A quadratic evaluated to the page edge runs away.
    curve = np.zeros(width, dtype=np.float64)
    left = int(np.floor(x_arr[0]))
    right = min(width - 1, int(np.ceil(x_arr[-1])))
    grid = np.arange(left, right + 1, dtype=np.float64)
    curve[left : right + 1] = np.interp(grid, x_arr, bow)
    return float(np.median(y_arr)), curve.astype(np.float32)


def _text_bows(
    found: list[tuple[float, np.ndarray]],
) -> list[tuple[float, np.ndarray]]:
    """Drop a line bowed far past the page's typical line."""
    if len(found) < 2:
        return found
    bows = np.asarray([float(np.max(np.abs(curve))) for _, curve in found], dtype=np.float64)
    limit = min(
        MAX_LINE_BOW_PX,
        max(float(np.median(bows)) * BOW_OUTLIER_RATIO, CURVATURE_CORRECT_PX),
    )
    return [sample for sample, bow in zip(found, bows, strict=True) if float(bow) <= limit]


def _centroids(band: np.ndarray, top: int) -> tuple[list[int], list[float]]:
    height, width = band.shape
    step = max(1, width // 40)
    rows = np.arange(height, dtype=np.float64)
    xs: list[int] = []
    ys: list[float] = []
    for x in range(0, width, step):
        column = band[:, x].astype(np.float64)
        total = float(column.sum())
        if total < 1:
            continue
        xs.append(x)
        ys.append(top + float(np.dot(rows, column) / total))
    return xs, ys


def _interpolate(centres: np.ndarray, curves: np.ndarray, height: int, width: int) -> np.ndarray:
    field = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        field[y] = _at(centres, curves, y, height)
    return field


def _at(centres: np.ndarray, curves: np.ndarray, y: int, height: int) -> np.ndarray:
    if y <= centres[0] or y >= centres[-1]:
        return _fade_edge(curves, y, centres, height)
    hi = int(np.searchsorted(centres, y))
    lo = max(hi - 1, 0)
    span = float(centres[hi] - centres[lo]) if hi < len(centres) else 1.0
    if span <= 0:
        return _row(curves, lo)
    weight = (y - float(centres[lo])) / span
    hi_row = curves[min(hi, len(curves) - 1)]
    return np.asarray((1.0 - weight) * curves[lo] + weight * hi_row, dtype=np.float32)


def _fade_edge(curves: np.ndarray, y: int, centres: np.ndarray, height: int) -> np.ndarray:
    if y <= centres[0]:
        weight = 0.0 if centres[0] <= 0 else y / float(centres[0])
        return _row(curves, 0) * weight
    span = float(height - 1 - centres[-1])
    weight = 0.0 if span <= 0 else (height - 1 - y) / span
    return _row(curves, -1) * weight


def _row(curves: np.ndarray, index: int) -> np.ndarray:
    return np.asarray(curves[index], dtype=np.float32)
