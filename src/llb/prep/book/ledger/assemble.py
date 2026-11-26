"""Merge engine proposals into one page ledger.

One engine's boxes cluster by near-equality, so a line engine's neighbours stay separate.
A layout region and a line engine are a different case: each line joins the tightest region
that owns it, and the block order is the column reading order. A page-sized box that only
wraps those regions is not itself a paragraph.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from llb.core.contracts.data_prep.book_blocks import (
    BLOCK_KINDS,
    Block,
    BlockKind,
    PageBlockLedger,
    RawReading,
    Region,
)
from llb.prep.book.constants import BLOCK_LEDGER_SCHEMA_ID, CONTRACT_VERSION
from llb.prep.book.engines.protocol import EngineRead, Proposed
from llb.prep.book.ledger.ids import block_id
from llb.prep.book.ledger.reading import STRUCTURAL_STAGES, recover
from llb.prep.book.ledger.reconcile import reconcile

# Cross-engine: intersection over the smaller box. Half the smaller region inside the other
# is the same block (a line inside a layout region). Same engine and variant: intersection over
# the union, so neighbouring lines and a small box inside a loose one stay apart.
OVERLAP_MIN = 0.5
# A more specific kind wins the block's label when a line and a formula share a region.
KIND_RANK: dict[str, int] = {
    "formula": 80,
    "table": 70,
    "figure": 60,
    "heading": 50,
    "caption": 40,
    "footnote": 30,
    "page-number": 20,
    "running-header": 20,
    "ornament": 10,
    "neighbour-page-bleed": 10,
    "body": 5,
    "unknown": 0,
}


@dataclass(frozen=True)
class _Tagged:
    stage: str
    engine: str
    model: str | None
    variant: str
    proposed: Proposed


@dataclass(frozen=True)
class _Assembly:
    members: tuple[_Tagged, ...]
    paragraph: bool
    kind: str | None


def assemble(
    page_id: str,
    pdf_page: int,
    width: int,
    height: int,
    reads: Sequence[EngineRead],
) -> PageBlockLedger:
    """The block ledger for one page. Ids follow geometry, not the order engines finished in."""
    tagged = [
        _Tagged(read.stage, read.engine, read.model, read.variant, proposed)
        for read in reads
        for proposed in read.blocks
    ]
    built: list[tuple[_Assembly, Region]] = []
    for item in _assemblies(tagged, height):
        region = _fit(_box_of(item.members), width, height)
        if region is not None:
            built.append((item, region))
    blocks = [
        _block(page_id, pdf_page, ordinal, item, region)
        for ordinal, (item, region) in enumerate(built)
    ]
    return PageBlockLedger(
        schema_id=BLOCK_LEDGER_SCHEMA_ID,
        schema_version=CONTRACT_VERSION,
        kind="book-ocr-block-ledger",
        page_id=page_id,
        pdf_page=pdf_page,
        blocks=blocks,
    )


def _assemblies(tagged: Sequence[_Tagged], height: int) -> list[_Assembly]:
    if not _mixed(tagged):
        ordered = sorted(cluster(tagged), key=_cluster_key)
        return [_Assembly(tuple(group), False, None) for group in ordered]
    recovered = recover(tagged, float(height))
    return [
        _Assembly(tuple(tagged[index] for index in members), paragraph, kind)
        for members, paragraph, kind in recovered
    ]


def _mixed(tagged: Sequence[_Tagged]) -> bool:
    stages = {item.stage for item in tagged}
    return bool(stages & STRUCTURAL_STAGES) and bool(stages - STRUCTURAL_STAGES)


def cluster(items: Sequence[_Tagged]) -> list[list[_Tagged]]:
    """Greedy clusters. Equal overlap keeps the earlier cluster, so the result does not depend on ties."""
    groups: list[list[_Tagged]] = []
    for item in _proposal_order(items):
        index = _best_group(groups, item)
        if index is None:
            groups.append([item])
        else:
            groups[index].append(item)
    return groups


def _proposal_order(items: Sequence[_Tagged]) -> list[_Tagged]:
    return sorted(items, key=lambda item: _proposal_key(item))


def _proposal_key(item: _Tagged) -> tuple[int, float, float, str, str]:
    box = item.proposed
    return (box.order, box.y0, box.x0, item.engine, item.variant)


def _best_group(groups: Sequence[Sequence[_Tagged]], item: _Tagged) -> int | None:
    found: tuple[float, int] | None = None
    for index, group in enumerate(groups):
        score = max(_pair_score(item, member) for member in group)
        if score < OVERLAP_MIN:
            continue
        if found is None or score > found[0]:
            found = (score, index)
    return None if found is None else found[1]


def _pair_score(left: _Tagged, right: _Tagged) -> float:
    if left.engine == right.engine and left.variant == right.variant:
        return _iou(left.proposed, right.proposed)
    return _overlap(left.proposed, right.proposed)


def _overlap(left: Proposed, right: Proposed) -> float:
    inter = _intersection(left, right)
    smaller = min(_area(left), _area(right))
    if inter <= 0 or smaller <= 0:
        return 0.0
    return inter / smaller


def _iou(left: Proposed, right: Proposed) -> float:
    inter = _intersection(left, right)
    union = _area(left) + _area(right) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _intersection(left: Proposed, right: Proposed) -> float:
    width = min(left.x1, right.x1) - max(left.x0, right.x0)
    height = min(left.y1, right.y1) - max(left.y0, right.y0)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _area(box: Proposed) -> float:
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)


def _cluster_key(group: Sequence[_Tagged]) -> tuple[int, float, float]:
    x0, y0, _, _ = _union(group)
    return (min(item.proposed.order for item in group), y0, x0)


def _union(group: Sequence[_Tagged]) -> tuple[float, float, float, float]:
    return (
        min(item.proposed.x0 for item in group),
        min(item.proposed.y0 for item in group),
        max(item.proposed.x1 for item in group),
        max(item.proposed.y1 for item in group),
    )


def _fit(box: tuple[float, float, float, float], width: int, height: int) -> Region | None:
    x0, y0 = max(0.0, box[0]), max(0.0, box[1])
    x1, y1 = min(float(width), box[2]), min(float(height), box[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return Region(x0=x0, y0=y0, x1=x1, y1=y1, width=width, height=height)


def _box_of(group: Sequence[_Tagged]) -> tuple[float, float, float, float]:
    structural = [item for item in group if item.stage in STRUCTURAL_STAGES]
    return _union(structural or group)


def _block(page_id: str, pdf_page: int, ordinal: int, item: _Assembly, region: Region) -> Block:
    readings = _readings(item.members)
    return Block(
        block_id=block_id(page_id, ordinal),
        pdf_page=pdf_page,
        ordinal=ordinal,
        kind=_chosen_kind(item),
        region=region,
        reading_order=ordinal,
        readings=readings,
        uncertainty=reconcile(readings),
        paragraph=item.paragraph,
    )


def _chosen_kind(item: _Assembly) -> BlockKind:
    if item.kind is not None:
        for known in BLOCK_KINDS:
            if item.kind == known:
                return known
    return _kind(item.members)


def _reading_key(item: _Tagged) -> tuple[str, str, float, float, int]:
    box = item.proposed
    return (item.engine, item.variant or "", box.y0, box.x0, box.order)


def _kind(group: Sequence[_Tagged]) -> BlockKind:
    winner = max(group, key=lambda item: (KIND_RANK.get(item.proposed.kind, 0), item.engine))
    for known in BLOCK_KINDS:
        if winner.proposed.kind == known:
            return known
    return "unknown"


def _readings(group: Sequence[_Tagged]) -> list[RawReading]:
    structural = {(item.engine, item.variant) for item in group if item.stage in STRUCTURAL_STAGES}
    buckets: dict[tuple[str, str, str | None], list[_Tagged]] = {}
    ordered = sorted(group, key=_reading_key)
    for item in ordered:
        if item.stage not in STRUCTURAL_STAGES and (item.engine, item.variant) in structural:
            continue
        buckets.setdefault((item.engine, item.variant, item.model), []).append(item)
    readings = []
    for (engine, variant, model), parts in buckets.items():
        texts = [part.proposed.text for part in parts if part.proposed.text]
        scores = [
            part.proposed.confidence for part in parts if part.proposed.confidence is not None
        ]
        readings.append(
            RawReading(
                engine=engine,
                model=model,
                variant=variant,
                text="\n".join(texts),
                confidence=None if not scores else sum(scores) / len(scores),
            )
        )
    return readings
