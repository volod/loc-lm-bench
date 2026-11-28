"""The printed references of each projected document.

The scan is the vault's. A resolved citation keeps the cited token's span here and the
target block's span in the document that holds that block. Ambiguous keys and mapping
gaps stay unresolved and name no target.
"""

import json
from typing import Literal

from llb.core.contracts.data_prep.book_references import (
    BookReferences,
    HierarchyPlace,
    IndexEntry,
    IndexPage,
    PrintedCitation,
    TechnicalBlock,
)
from llb.core.contracts.data_prep.book_structure import DocumentStructure, StructureNode
from llb.prep.book.project.locations import (
    ProjectionLocations,
    locate_blocks,
    lookup_vault_target,
    printed_blocks,
    reference_target,
)
from llb.prep.book.project.write import ProjectedDocument
from llb.prep.book.project.technical import TechnicalIdentity
from llb.prep.book.render.catalog import Catalog
from llb.prep.book.render.grammar import ReferenceGrammar
from llb.prep.book.render.index_terms import index_lines
from llb.prep.book.render.link import Cited, citation_hits
from llb.prep.book.render.names import parents_of, walk
from llb.prep.book.structure.text import norm_printed


def reference_sidecars(
    items: list[tuple[ProjectedDocument, str]],
    structure: DocumentStructure,
    catalog: Catalog,
    grammar: ReferenceGrammar,
    technical: dict[str, TechnicalIdentity],
) -> list[str]:
    """One JSON sidecar per document, in the same order as ``items``."""
    located = locate_blocks(items)
    pages = _observed(structure, catalog, located)
    places = _places(structure)
    return [
        _dump(_record(item, doc_id, catalog, grammar, located, pages, places, technical))
        for item, doc_id in items
    ]


def _record(
    item: ProjectedDocument,
    doc_id: str,
    catalog: Catalog,
    grammar: ReferenceGrammar,
    located: ProjectionLocations,
    pages: dict[str, int],
    places: dict[str, list[HierarchyPlace]],
    technical: dict[str, TechnicalIdentity],
) -> BookReferences:
    index = item.stem in catalog.index_stems
    return BookReferences(
        doc_id=doc_id,
        grammar=grammar.name,
        hierarchy=places.get(item.node_id, []),
        citations=_citations(item, catalog, grammar, located, index),
        index=_index(item.text, pages) if index else [],
        technical_blocks=_technical_blocks(item, doc_id, technical, located),
    )


def _citations(
    item: ProjectedDocument,
    catalog: Catalog,
    grammar: ReferenceGrammar,
    located: ProjectionLocations,
    index: bool,
) -> list[PrintedCitation]:
    hits = citation_hits(item.text, here=item.stem, catalog=catalog, grammar=grammar, index=index)
    generated = [
        (int(raw["char_start"]), int(raw["char_end"]))
        for page in item.pages
        for raw in page.blocks
        if raw.get("generated")
    ]
    rows = [
        _citation(hit, item.text, located)
        for hit in hits
        if not any(start < hit.end and hit.start < end for start, end in generated)
    ]
    rows.sort(key=lambda row: (row.cue_start, row.cue_end, row.kind, row.key))
    return rows


def _citation(hit: Cited, text: str, located: ProjectionLocations) -> PrintedCitation:
    _span(text, hit.start, hit.end, "citation cue")
    block = (
        lookup_vault_target(hit.target, located)
        if hit.outcome == "resolved" and hit.target
        else None
    )
    if block is not None:
        _span(located.text[block.doc_id], block.char_start, block.char_end, "citation target")
        return PrintedCitation(
            kind=hit.kind,
            key=hit.key,
            outcome="resolved",
            reason="",
            cue_start=hit.start,
            cue_end=hit.end,
            target=reference_target(block),
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
        for label in line.pages:
            pdf = pages.get(label.printed)
            if pdf is None:
                missing.append(label.printed)
            else:
                mapped.append(
                    IndexPage(
                        printed=label.printed,
                        pdf_page=pdf,
                        char_start=label.start,
                        char_end=label.end,
                    )
                )
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


def _technical_blocks(
    item: ProjectedDocument,
    doc_id: str,
    identities: dict[str, TechnicalIdentity],
    located: ProjectionLocations,
) -> list[TechnicalBlock]:
    blocks = []
    for _, _, block in printed_blocks(item, doc_id):
        if (identity := identities.get(block.block_id)) is not None:
            blocks.append(
                TechnicalBlock(
                    kind=identity.kind,
                    span=reference_target(block),
                    definition=reference_target(
                        block
                        if identity.defining_block_id == block.block_id
                        else located.by_id.get(identity.defining_block_id, block)
                    ),
                )
            )
    return blocks


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
                    heading_block_id=current.block_id,
                )
            )
            current = parents.get(current.node_id)
        chain.reverse()
        places[node_id] = chain
    return places


def _observed(
    structure: DocumentStructure, catalog: Catalog, located: ProjectionLocations
) -> dict[str, int]:
    return {
        norm_printed(page.printed): page.pdf_page
        for page in structure.pages
        if page.printed
        and page.status == "observed"
        and lookup_vault_target(catalog.pages.get(norm_printed(page.printed)), located) is not None
    }


def _span(text: str, start: int, end: int, label: str) -> None:
    if not (0 <= start < end <= len(text)) or not text[start:end].strip():
        raise ValueError(f"{label} span is outside the document")


def _dump(record: BookReferences) -> str:
    payload = record.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
