"""Which quarter turn makes the page's text upright.

A projection profile is enough when one axis wins by a wide margin, which is a page of
full-width lines turned on its side. A real page is settled from the letters: the bottoms of
the letters in a line share a baseline while the tops do not. Upside down, that is reversed. A
component counts as a letter only when the ring around it is clearly lighter than its ink, so
a photograph's texture does not vote.
"""

import cv2
import numpy as np

from llb.prep.book.preprocess.constants import (
    BAND_FRACTION,
    BASELINE_FLIP,
    BASELINE_LINE_GLYPHS,
    BASELINE_MIN_LINES,
    LINE_CENTER_GAP,
    PAPER_LIFT,
    PAPER_PAD,
    POLARITY_MARGIN,
    SIDEWAYS_RATIO,
)
from llb.prep.book.preprocess.geometry import rotate_upright
from llb.prep.book.preprocess.images import ink_mask, text_bands

_MIN_H = 8
_MAX_H = 36
_MIN_W = 3
_MAX_W = 40
_MIN_AREA = 20
_MAX_AREA = 700

Box = tuple[int, int, int, int]


def orientation_deg(grey: np.ndarray, *, placed_sideways: bool = False) -> int:
    """Degrees clockwise the content is turned from upright: 0, 90, 180, or 270.

    `placed_sideways` is the PDF image placement, not a guess at the content. When the page has
    enough letter lines, a negative baseline vote turns it over only if the scan was placed on its
    side: on a portrait page that vote also fires for an upright table. A page without those lines
    falls back to band polarity, which is what the synthetic fixtures are built to trip.
    """
    turned, _ = rotate_upright(grey, 90)
    if _sideways(grey, turned):
        return _facing(turned, 90, 270, allow_baseline=True)
    return _facing(grey, 0, 180, allow_baseline=placed_sideways)


def _sideways(grey: np.ndarray, turned: np.ndarray) -> bool:
    # Letter aspect is not used. A diagram on a white page has more wide pieces than tall
    # ones, and a page of small type can too. Full-width lines are the case that is unambiguous.
    return _axis_score(turned) > _axis_score(grey) * SIDEWAYS_RATIO


def _facing(levelled: np.ndarray, upright: int, flipped_angle: int, *, allow_baseline: bool) -> int:
    """`levelled` reads horizontally. Return the quarter turn that puts the baseline at the bottom."""
    vote, lines = _baseline(levelled)
    if lines >= BASELINE_MIN_LINES:
        # Real text. Band polarity is fooled by a figure or a ruled table, so it does not vote.
        if allow_baseline and vote < BASELINE_FLIP:
            return flipped_angle
        return upright
    flipped, _ = rotate_upright(levelled, 180)
    if _polarity(flipped) > _polarity(levelled) + POLARITY_MARGIN:
        return flipped_angle
    if _polarity(levelled) > _polarity(flipped) + POLARITY_MARGIN:
        return upright
    return upright


def _baseline(grey: np.ndarray) -> tuple[float, int]:
    """Median of per-line (top variance - bottom variance), and how many lines voted.

    Positive means the bottoms line up and the tops do not: the page is upright.
    """
    votes = [
        vote for vote in (_line_vote(line) for line in _lines(_letters(grey))) if vote is not None
    ]
    if not votes:
        return 0.0, 0
    return float(np.median(np.asarray(votes, dtype=np.float64))), len(votes)


def _line_vote(line: list[Box]) -> float | None:
    if len(line) < BASELINE_LINE_GLYPHS:
        return None
    tops = np.asarray([box[1] for box in line], dtype=np.float64)
    bottoms = np.asarray([box[1] + box[3] for box in line], dtype=np.float64)
    return float(np.var(tops) - np.var(bottoms))


def _lines(letters: list[Box]) -> list[list[Box]]:
    """Group letters whose centers share a baseline.

    Membership is against the line's own center, not the previous letter. Comparing with the
    previous letter lets a run of slightly lower letters walk down the page and swallow it.
    """
    ordered = sorted(letters, key=_center)
    if not ordered:
        return []
    lines: list[list[Box]] = [[ordered[0]]]
    for box in ordered[1:]:
        if _center(box) - _line_center(lines[-1]) > LINE_CENTER_GAP:
            lines.append([box])
        else:
            lines[-1].append(box)
    return lines


def _line_center(line: list[Box]) -> float:
    return float(np.median(np.asarray([_center(box) for box in line], dtype=np.float64)))


def _center(box: Box) -> float:
    return box[1] + box[3] / 2.0


def _letters(grey: np.ndarray) -> list[Box]:
    """Ink the size of a letter, sitting on a surround lighter than the ink itself."""
    ink = ink_mask(grey).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    found: list[Box] = []
    for index in range(1, count):
        box = _box(stats[index])
        if box is not None and _on_paper(grey, box):
            found.append(box)
    return found


def _box(stats: np.ndarray) -> Box | None:
    width = int(stats[cv2.CC_STAT_WIDTH])
    height = int(stats[cv2.CC_STAT_HEIGHT])
    area = int(stats[cv2.CC_STAT_AREA])
    if not (
        _MIN_H <= height <= _MAX_H and _MIN_W <= width <= _MAX_W and _MIN_AREA <= area <= _MAX_AREA
    ):
        return None
    return int(stats[cv2.CC_STAT_LEFT]), int(stats[cv2.CC_STAT_TOP]), width, height


def _on_paper(grey: np.ndarray, box: Box) -> bool:
    x, y, width, height = box
    page_h, page_w = grey.shape[:2]
    x0, y0 = max(0, x - PAPER_PAD), max(0, y - PAPER_PAD)
    x1, y1 = min(page_w, x + width + PAPER_PAD), min(page_h, y + height + PAPER_PAD)
    ring = np.array(grey[y0:y1, x0:x1], copy=True)
    ring[y - y0 : y - y0 + height, x - x0 : x - x0 + width] = 255
    ink = int(np.median(grey[y : y + height, x : x + width]))
    return int(np.median(ring)) >= ink + PAPER_LIFT


def _axis_score(grey: np.ndarray) -> float:
    rows = ink_mask(grey).sum(axis=1).astype(np.float64)
    if rows.size == 0 or float(rows.max()) <= 0:
        return 0.0
    return float(np.var(rows))


def _polarity(grey: np.ndarray) -> float:
    """Positive when the darker half of each text band is the lower one."""
    ink = ink_mask(grey)
    scores = [
        score
        for y0, y1 in text_bands(ink, BAND_FRACTION)
        if (score := _band_polarity(grey, y0, y1)) is not None
    ]
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _band_polarity(grey: np.ndarray, y0: int, y1: int) -> float | None:
    if y1 - y0 < 4:
        return None
    mid = (y0 + y1) // 2
    return float(grey[y0:mid].mean()) - float(grey[mid:y1].mean())
