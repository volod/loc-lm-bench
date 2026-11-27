"""Drop repeated badges and isolated specks, and join the pieces of one drawing."""

from llb.prep.book.assets.boxes import (
    Box,
    area,
    edge_gap,
    from_region,
    iou,
    normalized,
    span_limit,
    unite,
)
from llb.prep.book.assets.constants import (
    BADGE_IOU,
    BADGE_MAX_AREA_FRAC,
    BADGE_MAX_SIDE_FRAC,
    BADGE_REPEAT_PAGES,
    GROUP_GAP_FRAC,
    GROUP_GAP_MAX_PX,
    GROUP_GAP_MIN_PX,
    NOISE_MAX_AREA_FRAC,
    NOISE_MAX_SIDE_FRAC,
)
from llb.prep.book.assets.types import FigureGroup, Piece, piece_key
from llb.prep.book.render.catalog import canonical


class _Union:
    """Disjoint sets of block ids. Union is unconditional; the caller decides when."""

    def __init__(self, ids: list[str]) -> None:
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def members(self, root: str) -> list[str]:
        return [item for item in self.parent if self.find(item) == root]


def repeated_ornament_ids(pieces: list[Piece]) -> set[str]:
    """Small figure boxes that sit in the same place on two or more pages."""
    small = [piece for piece in pieces if _badge_sized(piece)]
    if len(small) < 2:
        return set()
    joined = _Union([piece.block.block_id for piece in small])
    _join_badges(joined, small)
    return _repeated(joined, small)


def group_pieces(pieces: list[Piece]) -> list[FigureGroup]:
    """Adjacent pieces with no conflicting printed number become one drawing."""
    pages: dict[int, list[Piece]] = {}
    for piece in pieces:
        pages.setdefault(piece.block.pdf_page, []).append(piece)
    groups: list[FigureGroup] = []
    for page in sorted(pages):
        groups.extend(_group_page(pages[page]))
    return groups


def drop_noise(groups: list[FigureGroup]) -> tuple[list[FigureGroup], set[str]]:
    """Groups that are specks with no number and no caption."""
    kept: list[FigureGroup] = []
    noise: set[str] = set()
    for group in groups:
        if _isolated(group):
            noise.update(piece.block.block_id for piece in group.pieces)
        else:
            kept.append(group)
    return kept, noise


def _badge_sized(piece: Piece) -> bool:
    region = piece.block.region
    if region is None:
        return False
    short = min(region.width, region.height)
    if short <= 0:
        return False
    box = from_region(region)
    side = max(box.x1 - box.x0, box.y1 - box.y0) / float(short)
    frac = area(box) / float(region.width * region.height)
    return side <= BADGE_MAX_SIDE_FRAC and frac <= BADGE_MAX_AREA_FRAC


def _join_badges(joined: _Union, small: list[Piece]) -> None:
    ordered = sorted(small, key=lambda piece: (piece.block.pdf_page, piece.block.block_id))
    for index, left in enumerate(ordered):
        _join_later(joined, left, ordered[index + 1 :])


def _join_later(joined: _Union, left: Piece, later: list[Piece]) -> None:
    for right in later:
        if left.block.pdf_page == right.block.pdf_page:
            continue
        if _same_place(left, right):
            joined.union(left.block.block_id, right.block.block_id)


def _same_place(left: Piece, right: Piece) -> bool:
    left_region = left.block.region
    right_region = right.block.region
    if left_region is None or right_region is None:
        return False
    return iou(normalized(left_region), normalized(right_region)) >= BADGE_IOU


def _repeated(joined: _Union, small: list[Piece]) -> set[str]:
    pages: dict[str, set[int]] = {}
    ids: dict[str, list[str]] = {}
    for piece in small:
        root = joined.find(piece.block.block_id)
        pages.setdefault(root, set()).add(piece.block.pdf_page)
        ids.setdefault(root, []).append(piece.block.block_id)
    rejected: set[str] = set()
    for root, found in pages.items():
        if len(found) >= BADGE_REPEAT_PAGES:
            rejected.update(ids[root])
    return rejected


def _group_page(pieces: list[Piece]) -> list[FigureGroup]:
    if not pieces:
        return []
    by_id = {piece.block.block_id: piece for piece in pieces}
    limit = _group_limit(pieces[0])
    joined = _Union(list(by_id))
    for left_id, right_id in _close_pairs(pieces, limit):
        _try_union(joined, by_id, left_id, right_id, limit)
    return _components(joined, pieces)


def _group_limit(piece: Piece) -> float:
    region = piece.block.region
    if region is None:
        return GROUP_GAP_MIN_PX
    return span_limit(
        region.width, region.height, GROUP_GAP_FRAC, GROUP_GAP_MIN_PX, GROUP_GAP_MAX_PX
    )


def _close_pairs(pieces: list[Piece], limit: float) -> list[tuple[str, str]]:
    found: list[tuple[float, str, str]] = []
    ordered = sorted(pieces, key=piece_key)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            gap = _piece_gap(left, right)
            if gap <= limit:
                found.append((gap, left.block.block_id, right.block.block_id))
    found.sort()
    return [(left_id, right_id) for _gap, left_id, right_id in found]


def _piece_gap(left: Piece, right: Piece) -> float:
    left_region = left.block.region
    right_region = right.block.region
    if left_region is None or right_region is None:
        return GROUP_GAP_MAX_PX + 1.0
    return edge_gap(from_region(left_region), from_region(right_region))


def _try_union(
    joined: _Union, by_id: dict[str, Piece], left_id: str, right_id: str, limit: float
) -> None:
    left_root = joined.find(left_id)
    right_root = joined.find(right_id)
    if left_root == right_root:
        return
    left_ids = joined.members(left_root)
    right_ids = joined.members(right_root)
    if not _compatible(left_ids, right_ids, by_id):
        return
    if _component_gap(left_ids, right_ids, by_id) > limit:
        return
    joined.union(left_id, right_id)


def _compatible(left_ids: list[str], right_ids: list[str], by_id: dict[str, Piece]) -> bool:
    numbers: set[str] = set()
    for item in (*left_ids, *right_ids):
        number = by_id[item].number
        if number:
            numbers.add(canonical(number))
    return len(numbers) <= 1


def _component_gap(left_ids: list[str], right_ids: list[str], by_id: dict[str, Piece]) -> float:
    return edge_gap(unite(_boxes(left_ids, by_id)), unite(_boxes(right_ids, by_id)))


def _boxes(ids: list[str], by_id: dict[str, Piece]) -> list[Box]:
    found: list[Box] = []
    for item in ids:
        region = by_id[item].block.region
        if region is not None:
            found.append(from_region(region))
    if not found:
        raise ValueError("figure group has no region")
    return found


def _components(joined: _Union, pieces: list[Piece]) -> list[FigureGroup]:
    buckets: dict[str, list[Piece]] = {}
    for piece in pieces:
        root = joined.find(piece.block.block_id)
        buckets.setdefault(root, []).append(piece)
    groups = [_finish(members) for members in buckets.values()]
    groups.sort(key=lambda group: piece_key(group.pieces[0]))
    return groups


def _finish(members: list[Piece]) -> FigureGroup:
    ordered = sorted(members, key=piece_key)
    number = next((piece.number for piece in ordered if piece.number), None)
    caption = next((piece.caption_id for piece in ordered if piece.caption_id), None)
    return FigureGroup(ordered, number, caption)


def _isolated(group: FigureGroup) -> bool:
    if group.number or group.caption_id:
        return False
    region = group.primary.region
    if region is None:
        return False
    box = group.union
    short = float(min(region.width, region.height))
    side = max(box.x1 - box.x0, box.y1 - box.y0)
    if side > NOISE_MAX_SIDE_FRAC * short:
        return False
    return area(box) <= NOISE_MAX_AREA_FRAC * float(region.width * region.height)
