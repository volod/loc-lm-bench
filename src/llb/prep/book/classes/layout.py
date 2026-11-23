"""Columns, ruling lines, drawings, and margin marks. No character is read."""

import cv2
import numpy as np

from llb.prep.book.classes.content import bands, components

_BODY_TOP = 0.08
_BODY_BOTTOM = 0.92
_DRAW_MIN = 0.025
_DRAW_MAX = 0.18


def column_spans(ink: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    """How many text columns, and the horizontal span of each. At most four."""
    height, width = ink.shape
    y0, y1 = int(height * _BODY_TOP), int(height * _BODY_BOTTOM)
    projection = ink[y0:y1].mean(axis=0) if y1 > y0 else ink.mean(axis=0)
    window = max(3, width // 80)
    smooth = np.convolve(projection, np.ones(window) / window, mode="same")
    peak = float(smooth.max()) if smooth.size else 0.0
    if peak <= 0:
        return 1, [(0, width)]
    hot = smooth >= peak * 0.08
    spans = [(start, end) for start, end in bands(hot) if end - start >= width * 0.15]
    if len(spans) >= 2:
        return min(4, len(spans)), spans[:4]
    body = spans[0] if spans else (0, width)
    valleys = _valleys(smooth, body, peak, width)
    if not valleys:
        return 1, [body]
    return min(4, 1 + len(valleys)), _split(body, valleys)


def rule_grid(mask: np.ndarray, x_height: float) -> tuple[int, int, float]:
    """Longest even run of horizontal rules, the vertical-rule count, and that run's height share."""
    horizontal = _rule_boxes(mask, x_height, horizontal=True)
    vertical = _rule_boxes(mask, x_height, horizontal=False)
    run, span = _grid_run(horizontal, x_height, mask.shape[0])
    return run, len(vertical), span


def drawings(stats: np.ndarray, count: int, ink: np.ndarray, x_height: float) -> tuple[float, int]:
    """Area share of compact drawings, and how many letter-sized components sit on them."""
    height, width = ink.shape
    page = float(height * width)
    boxes = []
    area = 0.0
    for index in range(1, count):
        x_pos, y_pos, box_w, box_h, pixels = (int(value) for value in stats[index])
        if box_w < x_height * 10 or box_h < x_height * 10:
            continue
        box = box_w * box_h
        if not _DRAW_MIN * page <= box <= _DRAW_MAX * page:
            continue
        if not 0.5 <= box_h / float(box_w) <= 2.2:
            continue
        if pixels / float(box) > 0.14:
            continue
        boxes.append((x_pos, y_pos, box_w, box_h))
        area += box / page
    return area, _labels(ink, boxes, x_height)


def corner_count(stats: np.ndarray, count: int, shape: tuple[int, ...], x_height: float) -> int:
    """Glyph-sized components wholly inside a corner. A running header is wider than the window."""
    height, width = int(shape[0]), int(shape[1])
    box_w, box_h = int(width * 0.10), int(height * 0.07)
    windows = (
        (0, 0, box_w, box_h),
        (width - box_w, 0, width, box_h),
        (0, height - box_h, box_w, height),
        (width - box_w, height - box_h, width, height),
    )
    found = 0
    minimum = max(4.0, x_height * x_height * 0.25)
    for index in range(1, count):
        x_pos, y_pos, comp_w, comp_h, area = (int(value) for value in stats[index])
        if area < minimum or comp_h > x_height * 2.2 or comp_w > x_height * 6:
            continue
        if any(_holds(window, x_pos, y_pos, comp_w, comp_h) for window in windows):
            found += 1
    return found


def margin_ink(ink: np.ndarray) -> float:
    """The busier of the two outer margins. An empty margin scores zero."""
    _height, width = ink.shape
    band = max(2, int(width * 0.04))
    left = float(ink[:, :band].mean())
    right = float(ink[:, width - band :].mean())
    return max(left, right)


def _valleys(
    smooth: np.ndarray, body: tuple[int, int], peak: float, width: int
) -> list[tuple[int, int]]:
    left, right = body
    region = smooth[left:right]
    level = float(np.median(region)) if region.size else peak
    low = region < max(level * 0.22, peak * 0.05)
    found = []
    span = right - left
    for start, end in bands(low):
        if end - start < width * 0.012:
            continue
        if start < span * 0.18 or span - end < span * 0.18:
            continue
        found.append((left + start, left + end))
    return found


def _split(body: tuple[int, int], valleys: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cursor = body[0]
    spans = []
    for start, end in valleys:
        spans.append((cursor, start))
        cursor = end
    spans.append((cursor, body[1]))
    return spans


def _rule_boxes(
    mask: np.ndarray, x_height: float, *, horizontal: bool
) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    binary = mask.astype(np.uint8) * 255
    fraction = 0.12 if horizontal else 0.05
    length = max(8, int((width if horizontal else height) * fraction))
    shape = (length, 1) if horizontal else (1, length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) > 0
    count, stats, _centroids = components(opened)
    boxes = []
    thin = max(2.5, x_height * 0.55)
    for index in range(1, count):
        x_pos, y_pos, box_w, box_h, _area = (int(value) for value in stats[index])
        thickness = box_h if horizontal else box_w
        long_side = box_w if horizontal else box_h
        if thickness <= thin and long_side >= length:
            boxes.append((x_pos, y_pos, box_w, box_h))
    return boxes


def _grid_run(
    boxes: list[tuple[int, int, int, int]], x_height: float, height: int
) -> tuple[int, float]:
    ys = sorted(box[1] for box in boxes)
    if len(ys) < 2:
        return 0, 0.0
    best = current = 1
    origin = 0
    y0 = y1 = ys[0]
    low, high = 0.6 * x_height, 7.0 * x_height
    for index, (earlier, later) in enumerate(zip(ys, ys[1:])):
        gap = later - earlier
        if low <= gap <= high:
            current += 1
            if current > best:
                best = current
                y0, y1 = ys[origin], later
        else:
            current = 1
            origin = index + 1
    if best < 2:
        return best, 0.0
    return best, (y1 - y0) / float(height)


def _labels(ink: np.ndarray, boxes: list[tuple[int, int, int, int]], x_height: float) -> int:
    if not boxes:
        return 0
    count, stats, centroids = components(ink)
    found = 0
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not (
            x_height * 0.45 <= height <= x_height * 2.2 and area >= 4 and width <= x_height * 5
        ):
            continue
        cx, cy = float(centroids[index][0]), float(centroids[index][1])
        pad = 2.0 * x_height
        for x_pos, y_pos, box_w, box_h in boxes:
            if (
                x_pos - pad <= cx <= x_pos + box_w + pad
                and y_pos - pad <= cy <= y_pos + box_h + pad
            ):
                found += 1
                break
    return found


def _holds(
    window: tuple[int, int, int, int], x_pos: int, y_pos: int, comp_w: int, comp_h: int
) -> bool:
    left, top, right, bottom = window
    return left <= x_pos and top <= y_pos and x_pos + comp_w <= right and y_pos + comp_h <= bottom
