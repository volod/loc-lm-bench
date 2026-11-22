"""Equation-number lines and the footnote band. Both are gaps and ink, not characters."""

from typing import cast

import cv2
import numpy as np

from llb.prep.book.classes.constants import FORMULA_LEADER_GAP
from llb.prep.book.classes.content import bands, components

_BODY_TOP = 0.06
_BODY_BOTTOM = 0.94
_FOOT_START = 0.80


def equation_hits(
    ink: np.ndarray, light: np.ndarray, x_height: float, spans: list[tuple[int, int]]
) -> tuple[int, int]:
    """Lines whose largest gap leaves a short mark at the right, and the text lines considered.

    A displayed equation with a number at the right margin has an empty gap. A dot leader fills
    that gap with faint marks, so an index line does not count.
    """
    height, _width = ink.shape
    y0, y1 = int(height * _BODY_TOP), int(height * _BODY_BOTTOM)
    hits = lines = 0
    for left, right in spans:
        column = ink[y0:y1, left:right]
        faint = light[y0:y1, left:right]
        span = right - left
        if span < 30:
            continue
        for start, end in bands(np.asarray(column.any(axis=1))):
            if not 2 <= (end - start) <= x_height * 3.2:
                continue
            lines += 1
            if _separated_tail(
                np.asarray(column[start:end].any(axis=0)),
                np.asarray(faint[start:end]),
                span,
            ):
                hits += 1
    return hits, lines


def footnote_band(mask: np.ndarray, ink: np.ndarray, x_height: float) -> float:
    """Height share of the text under a short rule in the bottom fifth. Zero when that text is absent."""
    height, width = mask.shape
    start = int(height * _FOOT_START)
    for x_pos, y_pos, box_w, box_h in _rule_boxes(mask[start:], x_height):
        if not width * 0.10 <= box_w <= width * 0.32:
            continue
        below = ink[start + y_pos + box_h :]
        if below.shape[0] < int(x_height):
            continue
        if _glyphs(below, x_height) >= 10:
            return cast(float, below.shape[0]) / float(cast(int, height))
    return 0.0


def _separated_tail(row: np.ndarray, faint: np.ndarray, span: int) -> bool:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(row.tolist()):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(row)))
    if len(runs) < 2:
        return False
    gap, left, right = max(
        (
            (runs[index + 1][0] - runs[index][1], runs[index][1], runs[index + 1][0])
            for index in range(len(runs) - 1)
        ),
        key=lambda item: item[0],
    )
    if not (
        gap >= span * 0.10
        and left > span * 0.35
        and runs[-1][1] > span * 0.82
        and (span - right) < span * 0.22
    ):
        return False
    if right <= left:
        return False
    return float(faint[:, left:right].mean()) < FORMULA_LEADER_GAP


def _rule_boxes(mask: np.ndarray, x_height: float) -> list[tuple[int, int, int, int]]:
    width = int(mask.shape[1])
    binary = mask.astype(np.uint8) * 255
    length = max(8, int(width * 0.12))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) > 0
    count, stats, _centroids = components(opened)
    thin = max(2.5, x_height * 0.55)
    boxes = []
    for index in range(1, count):
        x_pos, y_pos, box_w, box_h, _area = (int(value) for value in stats[index])
        if box_h <= thin and box_w >= length and box_h >= 1:
            boxes.append((x_pos, y_pos, box_w, box_h))
    return boxes


def _glyphs(ink: np.ndarray, x_height: float) -> int:
    count, stats, _centroids = components(ink)
    found = 0
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if 3 <= area <= 300 and height <= x_height * 2.5 and width >= 1:
            found += 1
    return found
