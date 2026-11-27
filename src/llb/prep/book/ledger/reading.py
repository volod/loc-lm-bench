"""Reading order and paragraph ownership for one page.

A layout region owns a line when the smaller box sits mostly inside the overlap.
The tightest such region wins. A line that two regions claim equally stays its own
block, and a body region that holds both columns does not keep the other column's
lines. A page-sized box that only groups tighter regions is not itself a paragraph.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from llb.prep.book.engines.protocol import Proposed

STRUCTURAL_STAGES = frozenset({"layout", "formula", "table"})

# Intersection over the smaller box. Half of that box inside the other is ownership.
OWN_MIN = 0.5
# Two scores this close, on regions of similar area, do not pick a region.
RIVAL_GAP = 0.08
TIGHTER_RATIO = 1.5
# A body box that swallows this many tighter regions is a container, not a paragraph.
CHILD_COVER = 0.85
CONTAINER_GROWTH = 1.4
CONTAINER_CHILDREN = 2
# Share of the ink span on either side of the midpoint. A line crossing both is a span.
COLUMN_MARGIN = 0.05
COLUMN_MIN = 2
# A bare folio in this fraction of the page height, measured from either edge.
FOLIO_BAND = 0.12
_FOLIO = re.compile(r"^\d{1,4}$")
_SPLIT = frozenset({"body", "unknown"})
_ASIDE = frozenset({"aside", "sidebar", "text-box", "boxed", "boxed-text"})
_APART = frozenset(
    {
        "heading",
        "caption",
        "table",
        "formula",
        "figure",
        "footnote",
        "running-header",
        "page-number",
        "ornament",
        "neighbour-page-bleed",
        "unknown",
    }
)


@dataclass(frozen=True)
class Placed:
    """One region or one line, in the rendered page's pixel space."""

    kind: str
    order: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(frozen=True)
class Group:
    """One recovered block: a region, the lines it owns, and any rival regions."""

    region: int | None
    lines: tuple[int, ...]
    rivals: tuple[int, ...]
    paragraph: bool
    kind: str | None


class _Source(Protocol):
    """One engine proposal the ledger can place. The stage decides region versus line."""

    @property
    def stage(self) -> str:
        """The stage that produced the proposal. Layout, formula, and table are regions."""

    @property
    def proposed(self) -> Proposed:
        """The box and text, in rendered-page pixels."""


def recover(
    items: Sequence[_Source], page_height: float
) -> list[tuple[tuple[int, ...], bool, str | None]]:
    """Index groups for a mixed page. Each group is the region, then the lines it owns."""
    structural = [index for index, item in enumerate(items) if item.stage in STRUCTURAL_STAGES]
    lines = [index for index, item in enumerate(items) if item.stage not in STRUCTURAL_STAGES]
    groups = arrange(
        [_placed(items[index]) for index in structural],
        [_placed(items[index]) for index in lines],
        page_height=page_height,
    )
    built: list[tuple[tuple[int, ...], bool, str | None]] = []
    for group in groups:
        members: list[int] = []
        if group.region is not None:
            members.append(structural[group.region])
        members.extend(lines[index] for index in group.lines)
        if members:
            built.append((tuple(members), group.paragraph, group.kind))
    return built


def _placed(item: _Source) -> Placed:
    box = item.proposed
    return Placed(box.kind, box.order, box.x0, box.y0, box.x1, box.y1, box.text)


def arrange(
    regions: Sequence[Placed], lines: Sequence[Placed], *, page_height: float
) -> list[Group]:
    """Groups in reading order. Containers and interleaved page boxes are not paragraphs."""
    active = [index for index, region in enumerate(regions) if not _container(index, regions)]
    owned: dict[int, list[int]] = {index: [] for index in active}
    loose: list[tuple[int, tuple[int, ...]]] = []
    for index, line in enumerate(lines):
        match = _match(line, active, regions)
        if match is None:
            loose.append((index, ()))
        elif isinstance(match, tuple):
            loose.append((index, match))
        else:
            owned[match].append(index)
    groups: list[Group] = []
    boxes = [*regions, *lines]
    mid, margin = _axis(boxes)
    for index in active:
        groups.extend(_emit(index, owned[index], regions, lines, mid, margin))
    for index, rivals in loose:
        groups.append(_loose(index, rivals, lines, page_height))
    return _order(groups, regions, lines, mid, margin)


def _emit(
    index: int,
    line_ids: list[int],
    regions: Sequence[Placed],
    lines: Sequence[Placed],
    mid: float,
    margin: float,
) -> list[Group]:
    region = regions[index]
    chosen = [lines[line_id] for line_id in line_ids]
    wide = _kind(region.kind) in _SPLIT and _side(region, mid, margin) == "span"
    if wide and _both(chosen, mid, margin):
        return [_line_group(line_id) for line_id in _by_y(line_ids, lines)]
    kept: list[int] = []
    detached: list[int] = []
    region_side = _side(region, mid, margin)
    for line_id in line_ids:
        line = lines[line_id]
        other_side = _side(line, mid, margin) not in {region_side, "span"}
        # A line that only clips the gutter is the other column. One whose width
        # sits inside the region belongs to it, even when its center has crossed.
        foreign = (
            region_side in {"left", "right"} and other_side and _x_cover(line, region) < OWN_MIN
        )
        apart = _kind(line.kind) in _APART and _kind(region.kind) == "body"
        if foreign or apart:
            detached.append(line_id)
        else:
            kept.append(line_id)
    groups = [Group(index, tuple(_by_y(kept, lines)), (), True, None)]
    groups.extend(_line_group(line_id) for line_id in detached)
    return groups


def _loose(
    index: int, rivals: tuple[int, ...], lines: Sequence[Placed], page_height: float
) -> Group:
    return Group(None, (index,), rivals, False, _folio(lines[index], page_height))


def _line_group(index: int) -> Group:
    return Group(None, (index,), (), False, None)


def _match(
    line: Placed, active: Sequence[int], regions: Sequence[Placed]
) -> int | tuple[int, ...] | None:
    scored = [
        (_overlap(line, regions[index]), _area(regions[index]), index)
        for index in active
        if _overlap(line, regions[index]) >= OWN_MIN
    ]
    if not scored:
        return None
    best = max(item[0] for item in scored)
    rivals = [item for item in scored if item[0] >= best - RIVAL_GAP]
    rivals.sort(key=lambda item: item[1])
    if len(rivals) >= 2 and rivals[1][1] < rivals[0][1] * TIGHTER_RATIO:
        return tuple(item[2] for item in rivals)
    return rivals[0][2]


def _container(index: int, regions: Sequence[Placed]) -> bool:
    region = regions[index]
    if _kind(region.kind) not in _SPLIT:
        return False
    children = 0
    for other_index, other in enumerate(regions):
        if other_index == index or _area(region) < _area(other) * CONTAINER_GROWTH:
            continue
        if _cover(other, region) >= CHILD_COVER:
            children += 1
    return children >= CONTAINER_CHILDREN


def _order(
    groups: list[Group],
    regions: Sequence[Placed],
    lines: Sequence[Placed],
    mid: float,
    margin: float,
) -> list[Group]:
    reps = [(_rep(group, regions, lines), group) for group in groups]
    if not _two_columns([rep for rep, _group in reps], mid, margin):
        reps.sort(key=lambda item: (item[0].y0, item[0].x0, _tie(item[1], regions, lines)))
        return [group for _rep_box, group in reps]
    ordered = sorted(reps, key=lambda item: (item[0].y0, item[0].x0))
    result: list[Group] = []
    band: list[tuple[Placed, Group]] = []
    for rep, group in ordered:
        if _side(rep, mid, margin) == "span":
            result.extend(_flush(band, mid, margin))
            band = []
            result.append(group)
        else:
            band.append((rep, group))
    result.extend(_flush(band, mid, margin))
    return result


def _flush(band: list[tuple[Placed, Group]], mid: float, margin: float) -> list[Group]:
    left = [group for rep, group in band if _side(rep, mid, margin) == "left"]
    right = [group for rep, group in band if _side(rep, mid, margin) != "left"]
    return [*left, *right]


def _two_columns(boxes: Sequence[Placed], mid: float, margin: float) -> bool:
    return _both(boxes, mid, margin)


def _both(boxes: Sequence[Placed], mid: float, margin: float) -> bool:
    left = sum(1 for box in boxes if _side(box, mid, margin) == "left")
    right = sum(1 for box in boxes if _side(box, mid, margin) == "right")
    return left >= COLUMN_MIN and right >= COLUMN_MIN


def _rep(group: Group, regions: Sequence[Placed], lines: Sequence[Placed]) -> Placed:
    if group.region is not None:
        return regions[group.region]
    return _union([lines[index] for index in group.lines])


def _by_y(line_ids: Sequence[int], lines: Sequence[Placed]) -> list[int]:
    return sorted(
        line_ids, key=lambda index: (lines[index].y0, lines[index].x0, lines[index].order)
    )


def _tie(group: Group, regions: Sequence[Placed], lines: Sequence[Placed]) -> int:
    if group.region is not None:
        return regions[group.region].order
    return min(lines[index].order for index in group.lines)


def _folio(line: Placed, page_height: float) -> str | None:
    if page_height <= 0 or _FOLIO.match(line.text.strip()) is None:
        return None
    band = page_height * FOLIO_BAND
    if line.y0 >= page_height - band or line.y1 <= band:
        return "page-number"
    return None


def _kind(kind: str) -> str:
    if kind in _ASIDE:
        return "unknown"
    return kind


def _side(box: Placed, mid: float, margin: float) -> str:
    if box.x0 < mid - margin and box.x1 > mid + margin:
        return "span"
    center = (box.x0 + box.x1) / 2
    return "left" if center < mid else "right"


def _axis(boxes: Sequence[Placed]) -> tuple[float, float]:
    if not boxes:
        return 0.0, 1.0
    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    span = max(right - left, 1.0)
    return left + span / 2, span * COLUMN_MARGIN


def _union(boxes: Sequence[Placed]) -> Placed:
    head = boxes[0]
    return Placed(
        head.kind,
        head.order,
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
        head.text,
    )


def _overlap(left: Placed, right: Placed) -> float:
    width = min(left.x1, right.x1) - max(left.x0, right.x0)
    height = min(left.y1, right.y1) - max(left.y0, right.y0)
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min(_area(left), _area(right))
    return width * height / smaller if smaller else 0.0


def _x_cover(inner: Placed, outer: Placed) -> float:
    width = min(inner.x1, outer.x1) - max(inner.x0, outer.x0)
    if width <= 0:
        return 0.0
    return width / max(1.0, inner.x1 - inner.x0)


def _cover(inner: Placed, outer: Placed) -> float:
    width = min(inner.x1, outer.x1) - max(inner.x0, outer.x0)
    height = min(inner.y1, outer.y1) - max(inner.y0, outer.y0)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height / _area(inner)


def _area(box: Placed) -> float:
    return max(1.0, (box.x1 - box.x0) * (box.y1 - box.y0))
