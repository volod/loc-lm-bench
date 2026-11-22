"""Box overlap, block matching, and rank agreement of reading order.

Figure overlap is the mean intersection-over-union after a one-to-one match. A reference figure
with no match contributes zero. Rank agreement is the share of matched pairs whose relative order
is the same on both sides. Fewer than two matched blocks leaves the rank undefined.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from llb.prep.book.metrics.constants import MATCH_IOU


@dataclass(frozen=True)
class Box:
    """An axis-aligned region. ``None`` on a block means the lane did not express a box."""

    x0: float
    y0: float
    x1: float
    y1: float


def intersection_over_union(left: Box, right: Box) -> float:
    """Area of the overlap divided by the area of the union. Zero when the boxes miss."""
    ix0, iy0 = max(left.x0, right.x0), max(left.y0, right.y0)
    ix1, iy1 = min(left.x1, right.x1), min(left.y1, right.y1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if intersection == 0.0:
        return 0.0
    area = (left.x1 - left.x0) * (left.y1 - left.y0) + (right.x1 - right.x0) * (right.y1 - right.y0)
    return intersection / (area - intersection)


def match_indices(
    reference: Sequence[Box | None],
    hypothesis: Sequence[Box | None],
    *,
    minimum: float = MATCH_IOU,
) -> list[tuple[int, int]]:
    """Greedy one-to-one matches at or above ``minimum`` overlap, best overlap first."""
    pairs: list[tuple[float, int, int]] = []
    for left, ref_box in enumerate(reference):
        if ref_box is None:
            continue
        for right, hyp_box in enumerate(hypothesis):
            if hyp_box is None:
                continue
            overlap = intersection_over_union(ref_box, hyp_box)
            if overlap >= minimum:
                pairs.append((overlap, left, right))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, left, right in pairs:
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        matched.append((left, right))
    return matched


def rank_agreement(pairs: Sequence[tuple[int, int]]) -> float | None:
    """Share of pairs with the same relative order. None when fewer than two items compare."""
    if len(pairs) < 2:
        return None
    agree = 0
    total = 0
    for index, (left_order, left_rank) in enumerate(pairs):
        for right_order, right_rank in pairs[index + 1 :]:
            ref_delta = left_order - right_order
            hyp_delta = left_rank - right_rank
            if ref_delta == 0 or hyp_delta == 0:
                continue
            total += 1
            if (ref_delta > 0) == (hyp_delta > 0):
                agree += 1
    if total == 0:
        return None
    return agree / total


def nearest_links(
    figures: Sequence[tuple[str, Box]], captions: Sequence[tuple[str, Box]]
) -> dict[str, str]:
    """Each figure's nearest caption, one caption to one figure, closest pair first."""
    pairs: list[tuple[float, str, str]] = []
    for figure_id, figure in figures:
        for caption_id, caption in captions:
            dx = _center(figure)[0] - _center(caption)[0]
            dy = _center(figure)[1] - _center(caption)[1]
            pairs.append((dx * dx + dy * dy, figure_id, caption_id))
    pairs.sort()
    used_figures: set[str] = set()
    used_captions: set[str] = set()
    links: dict[str, str] = {}
    for _, figure_id, caption_id in pairs:
        if figure_id in used_figures or caption_id in used_captions:
            continue
        used_figures.add(figure_id)
        used_captions.add(caption_id)
        links[figure_id] = caption_id
    return links


def _center(box: Box) -> tuple[float, float]:
    return ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)


def mean_overlap(reference: Sequence[Box], matched: Sequence[tuple[int, float]]) -> float:
    """Mean overlap over every reference box. A box that was not matched contributes zero."""
    if not reference:
        return 0.0
    scores = dict(matched)
    return sum(scores.get(index, 0.0) for index in range(len(reference))) / len(reference)
