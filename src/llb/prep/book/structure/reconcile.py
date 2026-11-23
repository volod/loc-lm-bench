"""Match the printed contents to observed headings, and check running headers.

A contents entry with no heading is reported. It does not become a node.
"""

import re

from llb.core.contracts.data_prep.book_structure import (
    ConflictKind,
    RunningHeader,
    StructureConflict,
)
from llb.prep.book.structure.constants import (
    MATCH_SCORE,
    PAGE_TOLERANCE,
    TITLE_JACCARD,
    TITLE_MISMATCH,
)
from llb.prep.book.structure.contents import Entry
from llb.prep.book.structure.draft import Draft, add_evidence
from llb.prep.book.structure.text import jaccard, norm_space, printed_delta

_SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)+$")


def reconcile(
    entries: list[Entry],
    nodes: list[Draft],
    headers: list[RunningHeader],
    printed_of: dict[int, str | None],
) -> list[StructureConflict]:
    """Contents matches, header agreement, and every disagreement, in a stable order."""
    conflicts: list[StructureConflict] = []
    _match(entries, nodes, printed_of, conflicts)
    _headers(nodes, headers, conflicts)
    return _sorted(conflicts)


def unmatched_contents(entries: list[Entry]) -> list[StructureConflict]:
    """Every contents entry, when the book had no heading to match one against."""
    return _sorted(_missing(entries, set()))


def _match(
    entries: list[Entry],
    nodes: list[Draft],
    printed_of: dict[int, str | None],
    conflicts: list[StructureConflict],
) -> None:
    pairs = _pairs(entries, nodes, printed_of)
    used_entries: set[int] = set()
    used_nodes: set[int] = set()
    for _rank, entry_index, node_index in pairs:
        if entry_index in used_entries or node_index in used_nodes:
            continue
        used_entries.add(entry_index)
        used_nodes.add(node_index)
        _agree(entries[entry_index], nodes[node_index], printed_of, conflicts)
    if entries:
        conflicts.extend(_missing(entries, used_entries))
        conflicts.extend(_extra(nodes, used_nodes))


def _pairs(
    entries: list[Entry], nodes: list[Draft], printed_of: dict[int, str | None]
) -> list[tuple[int, int, int]]:
    pairs = [
        (-_score(entry, node, printed_of), entry_index, node_index)
        for entry_index, entry in enumerate(entries)
        for node_index, node in enumerate(nodes)
        if _score(entry, node, printed_of) >= MATCH_SCORE
    ]
    pairs.sort()
    return pairs


def _score(entry: Entry, node: Draft, printed_of: dict[int, str | None]) -> int:
    score = 0
    if entry.number and node.number and entry.number.lower() == node.number.lower():
        score += 4
    if jaccard(entry.title, node.title) >= TITLE_JACCARD:
        score += 2
    printed = printed_of.get(node.pdf_start) or ""
    if entry.printed and printed and entry.printed == printed:
        score += 2
    elif entry.printed and printed and _within(entry.printed, printed):
        score += 1
    return score


def _within(left: str, right: str) -> bool:
    delta = printed_delta(left, right)
    return delta is not None and delta <= PAGE_TOLERANCE


def _agree(
    entry: Entry,
    node: Draft,
    printed_of: dict[int, str | None],
    conflicts: list[StructureConflict],
) -> None:
    add_evidence(node, "contents")
    node.fallback = "numbered-heading" not in node.evidence and "contents" not in node.evidence
    _title_conflict(entry, node, conflicts)
    _page_conflict(entry, node, printed_of, conflicts)


def _title_conflict(entry: Entry, node: Draft, conflicts: list[StructureConflict]) -> None:
    same_number = entry.number and entry.number.lower() == node.number.lower()
    if not same_number or not entry.title or not node.title:
        return
    if jaccard(entry.title, node.title) < TITLE_MISMATCH:
        conflicts.append(_conflict("title-mismatch", entry, node, node.pdf_start))


def _page_conflict(
    entry: Entry,
    node: Draft,
    printed_of: dict[int, str | None],
    conflicts: list[StructureConflict],
) -> None:
    printed = printed_of.get(node.pdf_start) or ""
    delta = printed_delta(entry.printed, printed) if entry.printed and printed else None
    if delta is not None and delta > PAGE_TOLERANCE:
        conflicts.append(_conflict("page-mismatch", entry, node, node.pdf_start))


def _headers(
    nodes: list[Draft], headers: list[RunningHeader], conflicts: list[StructureConflict]
) -> None:
    for span in headers:
        number = _section_number(span.text)
        covering = [node for node in nodes if node.pdf_start <= span.pdf_start <= node.pdf_end]
        if number:
            _number_header(number, covering, span, conflicts)
            continue
        for node in covering:
            if jaccard(span.text, node.title) >= TITLE_JACCARD:
                add_evidence(node, "running-header")


def _number_header(
    number: str,
    covering: list[Draft],
    span: RunningHeader,
    conflicts: list[StructureConflict],
) -> None:
    numbered = [node for node in covering if node.number]
    if any(node.number == number for node in numbered):
        for node in numbered:
            if node.number == number:
                add_evidence(node, "running-header")
        return
    if not numbered:
        return
    conflicts.append(
        StructureConflict(
            kind="header-mismatch",
            number=number,
            heading_title=numbered[-1].title,
            pdf_page=span.pdf_start,
            detail="running header section number disagrees with the heading",
        )
    )


def _section_number(text: str) -> str:
    cleaned = norm_space(text)
    if _SECTION_NUMBER.match(cleaned):
        return cleaned
    return ""


def _missing(entries: list[Entry], used: set[int]) -> list[StructureConflict]:
    return [
        StructureConflict(
            kind="contents-without-heading",
            number=entry.number,
            contents_title=entry.title,
            printed=entry.printed,
            pdf_page=entry.pdf_page,
            detail="contents entry has no matching heading",
        )
        for index, entry in enumerate(entries)
        if index not in used
    ]


def _extra(nodes: list[Draft], used: set[int]) -> list[StructureConflict]:
    return [
        StructureConflict(
            kind="heading-without-contents",
            number=node.number,
            heading_title=node.title,
            pdf_page=node.pdf_start,
            detail="heading has no matching contents entry",
        )
        for index, node in enumerate(nodes)
        if index not in used
    ]


def _conflict(kind: ConflictKind, entry: Entry, node: Draft, pdf: int) -> StructureConflict:
    return StructureConflict(
        kind=kind,
        number=entry.number or node.number,
        contents_title=entry.title,
        heading_title=node.title,
        printed=entry.printed,
        pdf_page=pdf,
        detail=kind,
    )


def _sorted(conflicts: list[StructureConflict]) -> list[StructureConflict]:
    return sorted(conflicts, key=lambda item: (item.pdf_page or 0, item.kind, item.number))
