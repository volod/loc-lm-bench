"""The printed references of each projected document.

The scan is the vault's. A resolved citation keeps the cited token's span here and the
target block's span in the document that holds that block. Ambiguous keys and mapping
gaps stay unresolved and name no target.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal

from llb.core.contracts.data_prep.book_references import (
    BookReferences,
    HierarchyPlace,
    IndexEntry,
    IndexPage,
    PrintedCitation,
    ReferenceTarget,
)
from llb.core.contracts.data_prep.book_structure import DocumentStructure, StructureNode
from llb.prep.book.project.write import ProjectedDocument
from llb.prep.book.render.catalog import Catalog, Target
from llb.prep.book.render.constants import CONTENT_DIRNAME
from llb.prep.book.render.grammar import ReferenceGrammar
from llb.prep.book.render.index_terms import index_lines
from llb.prep.book.render.link import Cited, citation_hits
from llb.prep.book.render.names import parents_of, walk
from llb.prep.book.structure.text import norm_printed

_PAGE_ANCHOR = re.compile(r"^p(\d+)$")


@dataclass(frozen=True, slots=True)
class _Block:
    doc_id: str
    block_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class _Located:
    by_anchor: dict[tuple[str, str], _Block]
    by_page: dict[tuple[str, int], _Block]
    text: dict[str, str]


def reference_sidecars(
    items: list[tuple[ProjectedDocument, str]],
    structure: DocumentStructure,
    catalog: Catalog,
    grammar: ReferenceGrammar,
) -> list[str]:
    """One JSON sidecar per document, in the same order as ``items``."""
    located = _locate(items)
    pages = _observed(structure)
    places = _places(structure)
    return [
        _dump(_record(item, doc_id, catalog, grammar, located, pages, places))
        for item, doc_id in items
    ]


def _record(
    item: ProjectedDocument,
    doc_id: str,
    catalog: Catalog,
    grammar: ReferenceGrammar,
    located: _Located,
    pages: dict[str, int],
    places: dict[str, list[HierarchyPlace]],
) -> BookReferences:
    index = item.stem in catalog.index_stems
    return BookReferences(
        doc_id=doc_id,
        grammar=grammar.name,
        hierarchy=places.get(item.node_id, []),
        citations=_citations(item, catalog, grammar, located, index),
        index=_index(item.text, pages) if index else [],
    )


def _citations(
    item: ProjectedDocument,
    catalog: Catalog,
    grammar: ReferenceGrammar,
    located: _Located,
    index: bool,
) -> list[PrintedCitation]:
    hits = citation_hits(item.text, here=item.stem, catalog=catalog, grammar=grammar, index=index)
    rows = [_citation(hit, item.text, located) for hit in hits]
    rows.sort(key=lambda row: (row.cue_start, row.cue_end, row.kind, row.key))
    return rows


def _citation(hit: Cited, text: str, located: _Located) -> PrintedCitation:
    _span(text, hit.start, hit.end, "citation cue")
    block = _lookup(hit.target, located) if hit.outcome == "resolved" and hit.target else None
    if block is not None:
        _span(located.text[block.doc_id], block.char_start, block.char_end, "citation target")
        return PrintedCitation(
            kind=hit.kind,
            key=hit.key,
            outcome="resolved",
            reason="",
            cue_start=hit.start,
            cue_end=hit.end,
            target=ReferenceTarget(
                doc_id=block.doc_id,
                block_id=block.block_id,
                char_start=block.char_start,
                char_end=block.char_end,
            ),
        )
    reason = _reason(hit.outcome)
    return PrintedCitation(
        kind=hit.kind,
        key=hit.key,
        outcome="unresolved",
        reason=reason,
        cue_start=hit.start,
        cue_end=hit.end,
        target=None,
    )


def _reason(outcome: str) -> Literal["ambiguous", "gap", "absent"]:
    if outcome == "ambiguous":
        return "ambiguous"
    if outcome == "gap":
        return "gap"
    return "absent"


def _index(text: str, pages: dict[str, int]) -> list[IndexEntry]:
    rows: list[IndexEntry] = []
    for line in index_lines(text):
        _span(text, line.term_start, line.term_end, "index term")
        if line.parent_start is not None and line.parent_end is not None:
            _span(text, line.parent_start, line.parent_end, "index parent")
        mapped: list[IndexPage] = []
        missing: list[str] = []
        for printed in line.pages:
            pdf = pages.get(printed)
            if pdf is None:
                missing.append(printed)
            else:
                mapped.append(IndexPage(printed=printed, pdf_page=pdf))
        rows.append(
            IndexEntry(
                term_start=line.term_start,
                term_end=line.term_end,
                parent_start=line.parent_start,
                parent_end=line.parent_end,
                pages=mapped,
                unresolved=missing,
            )
        )
    return rows


def _lookup(target: Target | None, located: _Located) -> _Block | None:
    if target is None:
        return None
    match = _PAGE_ANCHOR.match(target.anchor)
    if match is not None:
        return located.by_page.get((target.stem, int(match.group(1))))
    vault = f"{CONTENT_DIRNAME}/{target.stem}.md"
    return located.by_anchor.get((vault, target.anchor))


def _locate(items: list[tuple[ProjectedDocument, str]]) -> _Located:
    by_anchor: dict[tuple[str, str], _Block] = {}
    by_page: dict[tuple[str, int], _Block] = {}
    text: dict[str, str] = {}
    for item, doc_id in items:
        text[doc_id] = item.text
        for page in item.pages:
            for raw in page.blocks:
                block = _block(doc_id, raw)
                if block is None:
                    continue
                anchor = raw.get("vault_anchor")
                vault = raw.get("vault_file")
                if isinstance(anchor, str) and isinstance(vault, str) and anchor:
                    by_anchor.setdefault((vault, anchor), block)
                by_page.setdefault((item.stem, page.page), block)
    return _Located(by_anchor, by_page, text)


def _block(doc_id: str, raw: object) -> _Block | None:
    if not isinstance(raw, dict):
        return None
    block_id = raw.get("block_id")
    start = raw.get("char_start")
    end = raw.get("char_end")
    if not isinstance(block_id, str) or not isinstance(start, int) or not isinstance(end, int):
        return None
    if end <= start or not block_id:
        return None
    return _Block(doc_id, block_id, start, end)


def _places(structure: DocumentStructure) -> dict[str, list[HierarchyPlace]]:
    nodes = {node.node_id: node for node in walk(structure.hierarchy)}
    parents = parents_of(structure.hierarchy)
    places: dict[str, list[HierarchyPlace]] = {}
    for node_id, node in nodes.items():
        chain: list[HierarchyPlace] = []
        current: StructureNode | None = node
        seen: set[str] = set()
        while current is not None and current.node_id not in seen:
            seen.add(current.node_id)
            chain.append(
                HierarchyPlace(
                    node_id=current.node_id,
                    level=current.level,
                    number=current.number,
                    title=current.title,
                )
            )
            current = parents.get(current.node_id)
        chain.reverse()
        places[node_id] = chain
    return places


def _observed(structure: DocumentStructure) -> dict[str, int]:
    found: dict[str, list[int]] = {}
    gaps: set[str] = set()
    for page in structure.pages:
        if not page.printed:
            continue
        key = norm_printed(page.printed)
        if page.status == "gap":
            gaps.add(key)
            continue
        if page.status == "observed":
            found.setdefault(key, []).append(page.pdf_page)
    return {
        key: pages[0] for key, pages in found.items() if key not in gaps and len(set(pages)) == 1
    }


def _span(text: str, start: int, end: int, label: str) -> None:
    if not (0 <= start < end <= len(text)) or not text[start:end].strip():
        raise ValueError(f"{label} span is outside the document")


def _dump(record: BookReferences) -> str:
    payload = record.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
