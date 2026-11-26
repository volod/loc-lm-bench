"""One region when several stages of the same engine wrote the same box.

A whole-page converter is asked once per recognition stage. The boxes match.
Keeping every copy would print the paragraph once per stage.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, TypeVar, cast

from llb.core.contracts.data_prep.book_blocks import TableReading
from llb.prep.book.engines.protocol import Proposed
from llb.prep.book.grid import formula_from_text, table_from_text

# Intersection over union. A line inside a larger region stays a line.
SAME_BOX = 0.85
_STRUCTURAL = ("layout", "formula", "table")


class _Item(Protocol):
    """One staged proposal. The ledger's tagged row satisfies this.

    Properties stay read-only so a frozen row matches the protocol.
    """

    @property
    def stage(self) -> str: ...

    @property
    def engine(self) -> str: ...

    @property
    def model(self) -> str | None: ...

    @property
    def variant(self) -> str: ...

    @property
    def proposed(self) -> Proposed: ...


_T = TypeVar("_T", bound=_Item)


def with_native(proposed: Proposed) -> Proposed:
    """Fill formula and table fields from text when the reading file omitted them."""
    formula = proposed.formula
    table = _table(proposed)
    if formula is None and proposed.kind in {"formula", "inline-formula"} and proposed.text.strip():
        formula = formula_from_text(proposed.text, display=proposed.kind != "inline-formula")
    if formula is proposed.formula and table is proposed.table:
        return proposed
    return replace(proposed, formula=formula, table=table)


def _table(proposed: Proposed) -> TableReading | None:
    """The native grid, or a parse of the text. HTML that grew a second header row replaces a one-row label."""
    table = proposed.table
    if proposed.kind != "table" or not proposed.text.strip():
        return table
    if table is not None and not (table.header_rows < 2 and "<table" in proposed.text.lower()):
        return table
    caption = None if table is None else table.caption
    refreshed = table_from_text(proposed.text, caption)
    if refreshed is None:
        return table
    if table is not None and refreshed.header_rows <= table.header_rows:
        return table
    return refreshed


def collapse_repeated(items: Sequence[_T]) -> list[_T]:
    """Merge same-engine copies of one box. Other engines stay separate."""
    groups: list[list[_T]] = []
    for item in items:
        index = _match(groups, item)
        if index is None:
            groups.append([item])
        else:
            groups[index].append(item)
    return [_merge(group) for group in groups]


def _match(groups: Sequence[Sequence[_T]], item: _T) -> int | None:
    found: tuple[float, int] | None = None
    for index, group in enumerate(groups):
        if item.engine != group[0].engine or item.variant != group[0].variant:
            continue
        score = max(_iou(item.proposed, member.proposed) for member in group)
        if score < SAME_BOX:
            continue
        if found is None or score > found[0]:
            found = (score, index)
    return None if found is None else found[1]


def _merge(group: Sequence[_T]) -> _T:
    from llb.prep.book.ledger.assemble import KIND_RANK, _Tagged

    natived = [with_native(item.proposed) for item in group]
    winner = max(
        range(len(group)),
        key=lambda index: (KIND_RANK.get(natived[index].kind, 0), -natived[index].order),
    )
    text = _joined(natived)
    formula = next((item.formula for item in natived if item.formula is not None), None)
    table = next((item.table for item in natived if item.table is not None), None)
    merged = with_native(replace(natived[winner], text=text, formula=formula, table=table))
    source = group[winner]
    return cast(_T, _Tagged(_stage(group), source.engine, source.model, source.variant, merged))


def _joined(proposed: Sequence[Proposed]) -> str:
    """Different texts on one box stay in reading order. Identical copies stay once."""
    texts: list[str] = []
    for item in sorted(proposed, key=lambda item: item.order):
        if item.text and (not texts or texts[-1] != item.text):
            texts.append(item.text)
    return "\n".join(texts)


def _stage(group: Sequence[_Item]) -> str:
    stages = [item.stage for item in group]
    for name in _STRUCTURAL:
        if name in stages:
            return name
    return stages[0]


def _iou(left: Proposed, right: Proposed) -> float:
    width = min(left.x1, right.x1) - max(left.x0, right.x0)
    height = min(left.y1, right.y1) - max(left.y0, right.y0)
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = _area(left) + _area(right) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _area(box: Proposed) -> float:
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)
