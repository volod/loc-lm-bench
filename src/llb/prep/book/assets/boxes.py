"""Axis-aligned boxes for figure grouping and caption distance."""

import math
from collections.abc import Iterable

from llb.core.contracts.data_prep.book_blocks import Region


class Box:
    """A rectangle in one coordinate frame. Origin is the top left."""

    __slots__ = ("x0", "x1", "y0", "y1")

    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


def from_region(region: Region) -> Box:
    """The ledger region as a box."""
    return Box(region.x0, region.y0, region.x1, region.y1)


def normalized(region: Region) -> Box:
    """The region as fractions of its page, so two scans can share a badge."""
    return Box(
        region.x0 / region.width,
        region.y0 / region.height,
        region.x1 / region.width,
        region.y1 / region.height,
    )


def unite(boxes: Iterable[Box]) -> Box:
    """The smallest box that contains every input."""
    found = list(boxes)
    if not found:
        raise ValueError("union of no boxes")
    return Box(
        min(box.x0 for box in found),
        min(box.y0 for box in found),
        max(box.x1 for box in found),
        max(box.y1 for box in found),
    )


def area(box: Box) -> float:
    """Width times height. Zero when the box is empty or inverted."""
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)


def center(box: Box) -> tuple[float, float]:
    """The box center."""
    return ((box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0)


def edge_gap(left: Box, right: Box) -> float:
    """Distance between the boxes. Zero when they touch or overlap."""
    dx = max(left.x0 - right.x1, right.x0 - left.x1, 0.0)
    dy = max(left.y0 - right.y1, right.y0 - left.y1, 0.0)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.hypot(dx, dy)


def iou(left: Box, right: Box) -> float:
    """Intersection over union. Zero when the boxes miss."""
    ix0, iy0 = max(left.x0, right.x0), max(left.y0, right.y0)
    ix1, iy1 = min(left.x1, right.x1), min(left.y1, right.y1)
    overlap = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if overlap == 0.0:
        return 0.0
    total = area(left) + area(right) - overlap
    if total <= 0.0:
        return 0.0
    return overlap / total


def span_limit(width: int, height: int, fraction: float, low: float, high: float) -> float:
    """``fraction`` of the shorter side, clamped to ``[low, high]`` pixels."""
    return min(high, max(low, fraction * float(min(width, height))))


def clip(box: Box, width: int, height: int) -> Box:
    """``box`` clipped to the page."""
    return Box(
        min(max(box.x0, 0.0), float(width)),
        min(max(box.y0, 0.0), float(height)),
        min(max(box.x1, 0.0), float(width)),
        min(max(box.y1, 0.0), float(height)),
    )


def pixel_box(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Integer pixel bounds of ``box``, clipped to the image. None when empty."""
    x0 = max(0, math.floor(box.x0))
    y0 = max(0, math.floor(box.y0))
    x1 = min(width, math.ceil(box.x1))
    y1 = min(height, math.ceil(box.y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1
