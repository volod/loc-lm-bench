"""Targets a reference may name, and the ones it must not.

A figure, table, or equation number is unique when one caption or display
number carries it. A page-footer echo of that same number is the running
index line, not a second item. Two captions with one number are ambiguous,
and so are several footer sightings with no caption. A printed page links
only when one observed page carries that label. A gap, or two claimants,
does not.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from llb.core.contracts.data_prep.book_structure import (
    DocumentStructure,
    NumberedItem,
    PrintedPage,
    StructureNode,
)
from llb.prep.book.render.grammar import ReferenceGrammar
from llb.prep.book.render.names import anchor_map, walk
from llb.prep.book.render.types import VaultFile
from llb.prep.book.structure.text import norm_printed

_ANCHOR = re.compile(r"^\^([A-Za-z][A-Za-z0-9-]*)\s*$")
_PAGE = re.compile(r"^p(\d+)$")
_FOOTER = "page-footer"


@dataclass(frozen=True, slots=True)
class Target:
    """One block anchor in one content note."""

    anchor: str
    stem: str


@dataclass(frozen=True, slots=True)
class ListRow:
    """One inventory item the generated list can point at."""

    number: str
    printed: str | None
    pdf_page: int
    anchor: str
    stem: str


@dataclass(frozen=True, slots=True)
class Catalog:
    """Lookups for one vault. Built after the notes exist and before they are linked."""

    unique: dict[tuple[str, str], Target]
    ambiguous: frozenset[tuple[str, str]]
    pages: dict[str, Target]
    page_ambiguous: frozenset[str]
    gaps: frozenset[str]
    rows: dict[str, tuple[ListRow, ...]]
    index_stems: frozenset[str]


def canonical(number: str) -> str:
    """``1-1`` and ``1.1`` share a key. A second item with that key is ambiguous."""
    return re.sub(r"[^a-z0-9]+", "-", number.strip().lower()).strip("-")


def build_catalog(
    files: list[VaultFile], structure: DocumentStructure, grammar: ReferenceGrammar
) -> Catalog:
    """The anchors the notes actually contain, keyed the way a citation looks them up."""
    anchors = _anchors(files)
    names = anchor_map(structure)
    unique, ambiguous, rows = _items(structure, grammar, names, anchors)
    node_unique, node_ambiguous = _nodes(structure, grammar, names, anchors)
    unique.update(node_unique)
    ambiguous = ambiguous | node_ambiguous
    pages, page_ambiguous, gaps = _pages(structure, anchors)
    return Catalog(
        unique=unique,
        ambiguous=ambiguous,
        pages=pages,
        page_ambiguous=page_ambiguous,
        gaps=gaps,
        rows=rows,
        index_stems=_index_stems(files, grammar),
    )


def _anchors(files: list[VaultFile]) -> dict[str, str]:
    found: dict[str, str] = {}
    for file in files:
        for line in file.body.splitlines():
            matched = _ANCHOR.match(line.strip())
            if matched and matched.group(1) not in found:
                found[matched.group(1)] = file.stem
    return found


def _items(
    structure: DocumentStructure,
    grammar: ReferenceGrammar,
    names: dict[str, str],
    anchors: dict[str, str],
) -> tuple[
    dict[tuple[str, str], Target], frozenset[tuple[str, str]], dict[str, tuple[ListRow, ...]]
]:
    wanted = {kind.name for kind in grammar.kinds if kind.inventory}
    grouped: dict[tuple[str, str], list[NumberedItem]] = defaultdict(list)
    for item in structure.inventory:
        if item.kind in wanted:
            grouped[(item.kind, canonical(item.number))].append(item)
    unique: dict[tuple[str, str], Target] = {}
    ambiguous: set[tuple[str, str]] = set()
    rows: dict[str, list[ListRow]] = defaultdict(list)
    for key, items in grouped.items():
        _one_item(key, items, names, anchors, unique, ambiguous, rows)
    frozen = {kind: tuple(_ordered(found)) for kind, found in rows.items()}
    return unique, frozenset(ambiguous), frozen


def _one_item(
    key: tuple[str, str],
    items: list[NumberedItem],
    names: dict[str, str],
    anchors: dict[str, str],
    unique: dict[tuple[str, str], Target],
    ambiguous: set[tuple[str, str]],
    rows: dict[str, list[ListRow]],
) -> None:
    defining = [item for item in items if item.evidence != _FOOTER]
    pool = defining or items
    located = [row for row in (_row(item, names, anchors) for item in pool) if row is not None]
    if len(defining) == 1 and located:
        unique[key] = Target(located[0].anchor, located[0].stem)
    elif not defining and len(items) == 1 and located:
        unique[key] = Target(located[0].anchor, located[0].stem)
    elif len(pool) > 1:
        ambiguous.add(key)
    if located and (defining or len(items) == 1):
        rows[key[0]].extend(located)


def _row(item: NumberedItem, names: dict[str, str], anchors: dict[str, str]) -> ListRow | None:
    anchor = names.get(item.block_id, "")
    stem = anchors.get(anchor, "")
    if not anchor or not stem:
        return None
    return ListRow(item.number, item.printed, item.pdf_page, anchor, stem)


def _ordered(rows: list[ListRow]) -> list[ListRow]:
    return sorted(rows, key=lambda row: (row.pdf_page, canonical(row.number), row.anchor))


def _nodes(
    structure: DocumentStructure,
    grammar: ReferenceGrammar,
    names: dict[str, str],
    anchors: dict[str, str],
) -> tuple[dict[tuple[str, str], Target], frozenset[tuple[str, str]]]:
    grouped = _group_nodes(structure, grammar)
    unique: dict[tuple[str, str], Target] = {}
    ambiguous: set[tuple[str, str]] = set()
    for key, nodes in grouped.items():
        _one_node(key, nodes, names, anchors, unique, ambiguous)
    return unique, frozenset(ambiguous)


def _group_nodes(
    structure: DocumentStructure, grammar: ReferenceGrammar
) -> dict[tuple[str, str], list[StructureNode]]:
    levels = {level: kind.name for kind in grammar.kinds for level in kind.levels}
    grouped: dict[tuple[str, str], list[StructureNode]] = defaultdict(list)
    for node in walk(structure.hierarchy):
        kind = levels.get(node.level, "")
        if kind and node.number and node.block_id:
            grouped[(kind, canonical(node.number))].append(node)
    return grouped


def _one_node(
    key: tuple[str, str],
    nodes: list[StructureNode],
    names: dict[str, str],
    anchors: dict[str, str],
    unique: dict[tuple[str, str], Target],
    ambiguous: set[tuple[str, str]],
) -> None:
    located = [
        target for target in (_node_target(node, names, anchors) for node in nodes) if target
    ]
    if len(nodes) == 1 and located:
        unique[key] = located[0]
    elif len(nodes) > 1:
        ambiguous.add(key)


def _node_target(
    node: StructureNode, names: dict[str, str], anchors: dict[str, str]
) -> Target | None:
    anchor = names.get(node.block_id or "", "")
    stem = anchors.get(anchor, "")
    if not anchor or not stem:
        return None
    return Target(anchor, stem)


def _pages(
    structure: DocumentStructure, anchors: dict[str, str]
) -> tuple[dict[str, Target], frozenset[str], frozenset[str]]:
    observed: dict[str, list[Target]] = defaultdict(list)
    gaps: set[str] = set()
    for page in structure.pages:
        _page(page, anchors, observed, gaps)
    unique, ambiguous = _unique_pages(observed, gaps)
    gap_only = frozenset(gaps - set(unique) - ambiguous)
    return unique, ambiguous, gap_only


def _page(
    page: PrintedPage,
    anchors: dict[str, str],
    observed: dict[str, list[Target]],
    gaps: set[str],
) -> None:
    if not page.printed:
        return
    key = norm_printed(page.printed)
    if page.status == "gap":
        gaps.add(key)
        return
    if page.status != "observed":
        return
    anchor = _page_anchor(page, anchors)
    stem = anchors.get(anchor, "") if anchor else ""
    if anchor and stem:
        observed[key].append(Target(anchor, stem))


def _page_anchor(page: PrintedPage, anchors: dict[str, str]) -> str | None:
    if page.page_id in anchors:
        return page.page_id
    for name in anchors:
        matched = _PAGE.match(name)
        if matched and int(matched.group(1)) == page.pdf_page:
            return name
    return None


def _unique_pages(
    observed: dict[str, list[Target]], gaps: set[str]
) -> tuple[dict[str, Target], frozenset[str]]:
    unique: dict[str, Target] = {}
    ambiguous: set[str] = set()
    for key, targets in observed.items():
        distinct = {(item.anchor, item.stem) for item in targets}
        if len(distinct) == 1 and key not in gaps:
            unique[key] = targets[0]
        else:
            ambiguous.add(key)
    return unique, frozenset(ambiguous)


def _index_stems(files: list[VaultFile], grammar: ReferenceGrammar) -> frozenset[str]:
    return frozenset(
        file.stem for file in files if file.node.title.casefold() in grammar.index_titles
    )
