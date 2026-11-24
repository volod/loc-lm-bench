"""A seeded, stratified pilot. A page is chosen once. An empty class is named, not silently dropped."""

import random
from collections.abc import Sequence
from typing import cast

from llb.core.contracts.data_prep.book_classes import (
    PAGE_CLASS_IDS,
    PAGE_CLASS_LABELS,
    PageClassId,
    PageClassRecord,
    PilotPageEntry,
    PilotSelection,
    PlacedSignals,
)
from llb.prep.book.classes.constants import PILOT_SCHEMA_ID, SCHEMA_VERSION

_HEAD = 4


def select_pilot(
    records: Sequence[PageClassRecord], *, source_sha256: str, seed: int, size: int
) -> PilotSelection:
    """`size` pages spread across the classes that have a member. `seed` picks inside each head."""
    _check(size, seed)
    unique = _unique(records)
    if not unique:
        raise ValueError("no classified pages to select from")
    members, present, empty = _index(unique)
    chosen = _choose(members, present, seed, size)
    return _assemble(chosen, present, empty, source_sha256=source_sha256, seed=seed, size=size)


def check_frozen(selection: PilotSelection, frozen: PilotSelection) -> None:
    """Refuse drift in the source, sampling parameters, class coverage, or page metadata."""
    actual = selection.model_dump()
    changed = [field for field, value in frozen.model_dump().items() if actual[field] != value]
    if changed:
        raise ValueError(f"frozen pilot differs in {', '.join(changed)}")


def _check(size: int, seed: int) -> None:
    if size < 1:
        raise ValueError("pilot size must be at least 1")
    if seed < 0:
        raise ValueError("pilot seed must be at least 0")


def _index(
    unique: Sequence[PageClassRecord],
) -> tuple[dict[str, list[PageClassRecord]], list[str], list[str]]:
    members: dict[str, list[PageClassRecord]] = {class_id: [] for class_id in PAGE_CLASS_IDS}
    for record in unique:
        for matched in record.classes:
            members[matched].append(record)
    present = [class_id for class_id in PAGE_CLASS_IDS if members[class_id]]
    empty = [PAGE_CLASS_LABELS[name] for name in PAGE_CLASS_IDS if name not in present]
    return members, present, empty


def _choose(
    members: dict[str, list[PageClassRecord]], present: list[str], seed: int, size: int
) -> dict[int, PilotPageEntry]:
    quota = _quotas(size, present)
    rng = random.Random(seed)
    chosen: dict[int, PilotPageEntry] = {}
    for name in present:
        _take(rng, members[name], name, quota[name], chosen)
    _fill(rng, members, present, size, chosen)
    return chosen


def _assemble(
    chosen: dict[int, PilotPageEntry],
    present: list[str],
    empty: list[str],
    *,
    source_sha256: str,
    seed: int,
    size: int,
) -> PilotSelection:
    pages = [chosen[number] for number in sorted(chosen)]
    covered = {page.represents for page in pages}
    unfilled = [PAGE_CLASS_LABELS[name] for name in present if name not in covered]
    coverage = {name: _count(pages, name) for name in PAGE_CLASS_IDS}
    return PilotSelection(
        schema_id=PILOT_SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        kind="book-ocr-pilot-selection",
        source_sha256=source_sha256,
        seed=seed,
        size=size,
        pages=pages,
        empty_classes=empty,
        unfilled_classes=unfilled,
        coverage=coverage,
    )


def _count(pages: list[PilotPageEntry], name: str) -> int:
    return sum(page.represents == name for page in pages)


def _unique(records: Sequence[PageClassRecord]) -> list[PageClassRecord]:
    by_page: dict[int, PageClassRecord] = {}
    for record in records:
        by_page.setdefault(record.pdf_page, record)
    return [by_page[number] for number in sorted(by_page)]


def _quotas(size: int, present: list[str]) -> dict[str, int]:
    quota = {name: 0 for name in present}
    if not present or size < 1:
        return quota
    if size < len(present):
        for name in present[:size]:
            quota[name] = 1
        return quota
    base, extra = divmod(size, len(present))
    for index, name in enumerate(present):
        quota[name] = base + (1 if index < extra else 0)
    return quota


def _take(
    rng: random.Random,
    pool: list[PageClassRecord],
    name: str,
    count: int,
    chosen: dict[int, PilotPageEntry],
) -> None:
    """Prefer pages whose routing class is `name`, then any other member, from the strongest head."""
    if count <= 0:
        return
    primary = _ranked(pool, name, chosen, primary=True)
    others = _ranked(pool, name, chosen, primary=False)
    picked = _sample_head(rng, primary, count)
    if len(picked) < count:
        picked.extend(_sample_head(rng, others, count - len(picked)))
    for record in picked:
        chosen[record.pdf_page] = _entry(record, name)


def _fill(
    rng: random.Random,
    members: dict[str, list[PageClassRecord]],
    present: list[str],
    size: int,
    chosen: dict[int, PilotPageEntry],
) -> None:
    """Spend leftover slots on classes that still have an unused page, in routing order."""
    while len(chosen) < size:
        grew = False
        for name in present:
            if len(chosen) >= size:
                return
            before = len(chosen)
            _take(rng, members[name], name, 1, chosen)
            grew = grew or len(chosen) > before
        if not grew:
            return


def _ranked(
    pool: list[PageClassRecord], name: str, chosen: dict[int, PilotPageEntry], *, primary: bool
) -> list[PageClassRecord]:
    free = [
        record
        for record in pool
        if record.pdf_page not in chosen and (record.primary == name) is primary
    ]
    return sorted(free, key=lambda record: (-_strength(record.signals, name), record.pdf_page))


def _sample_head(
    rng: random.Random, ranked: list[PageClassRecord], count: int
) -> list[PageClassRecord]:
    if count <= 0 or not ranked:
        return []
    head = ranked[: max(count, min(len(ranked), count * _HEAD))]
    if len(head) <= count:
        return list(head)
    return rng.sample(head, count)


def _strength(signals: PlacedSignals, name: str) -> float:
    if name == "severe_curvature":
        return signals.line_curvature_px
    if name == "dense_formulas":
        return signals.formula_density
    if name == "large_tables":
        return float(signals.ruling_lines)
    if name == "labelled_diagrams":
        return signals.figure_area
    if name == "small_footnotes":
        # A footnote rule sits near the bottom. A longer band is a separator higher on the page.
        return -signals.footnote_band
    if name == "gutter_corner":
        return float(signals.corner_components) + (1.0 if signals.has_bleed else 0.0)
    if name == "grey_background":
        return signals.background_unevenness
    if name == "multi_column":
        return float(signals.column_count)
    return 0.0


def _entry(record: PageClassRecord, represents: str) -> PilotPageEntry:
    return PilotPageEntry(
        pdf_page=record.pdf_page,
        page_id=record.page_id,
        represents=cast(PageClassId, represents),
        classes=list(record.classes),
        primary=record.primary,
        signals=record.signals,
    )
