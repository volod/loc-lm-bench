"""Infer `metadata/document_structure.json` from the block ledgers of one run."""

import logging
from collections import defaultdict
from pathlib import Path

from llb.core.contracts.data_prep.book_structure import (
    STRUCTURE_KIND,
    STRUCTURE_SCHEMA_ID,
    STRUCTURE_SCHEMA_VERSION,
    DocumentStructure,
)
from llb.prep.book.config import read_config
from llb.prep.book.structure.contents import contents_entries, contents_pages
from llb.prep.book.structure.furniture import header_spans, page_hits
from llb.prep.book.structure.headings import headings_of
from llb.prep.book.structure.inventory import inventory_of
from llb.prep.book.structure.io import structure_path, write_structure
from llb.prep.book.review.settled import terminal_failure
from llb.prep.book.structure.lines import Line, load_lines
from llb.prep.book.structure.numbering import ambiguous_folios, printed_pages
from llb.prep.book.structure.tree import build_hierarchy

_LOG = logging.getLogger(__name__)


def infer_structure(
    lines: list[Line], pages: list[int], page_count: int, source_sha256: str
) -> DocumentStructure:
    """The structure of these pages. `lines` may be empty; every page is still mapped."""
    grouped = _grouped(lines, pages)
    hits = page_hits(lines)
    mapping = printed_pages(pages, hits, page_count)
    printed_of = {page.pdf_page: page.printed for page in mapping}
    number_ids = {hit.block_id for hit in hits}
    headers = header_spans(lines, number_ids, pages)
    header_ids = {block_id for span in headers for block_id in span.block_ids}
    toc = contents_pages(grouped)
    entries = contents_entries(grouped, set(toc))
    headings = headings_of(grouped, set(toc), number_ids | header_ids)
    hierarchy, conflicts = build_hierarchy(pages, headings, entries, headers, printed_of)
    return DocumentStructure(
        schema_id=STRUCTURE_SCHEMA_ID,
        schema_version=STRUCTURE_SCHEMA_VERSION,
        kind=STRUCTURE_KIND,
        source_sha256=source_sha256,
        pages=mapping,
        ambiguous_folios=ambiguous_folios(hits),
        headers=headers,
        hierarchy=hierarchy,
        inventory=inventory_of(lines, printed_of),
        conflicts=conflicts,
        contents_pages=toc,
    )


def build_structure(run_dir: Path) -> DocumentStructure:
    """Write the structure of a run.

    A page that has not finished is refused. A page whose recognition failed
    terminally is included, with no lines.
    """
    config = read_config(run_dir)
    lines, missing = load_lines(run_dir, config)
    unfinished = [pid for pid in missing if not terminal_failure(run_dir, pid)]
    if unfinished:
        raise ValueError(f"block ledger has not finished for {_brief(unfinished)}")
    structure = infer_structure(
        lines, list(config.pages), config.source.page_count, config.source.sha256
    )
    write_structure(structure_path(run_dir), structure)
    _LOG.info(
        "[book-structure] %s pages, %s conflicts, %s inventory",
        len(structure.pages),
        len(structure.conflicts),
        len(structure.inventory),
    )
    return structure


def _grouped(lines: list[Line], pages: list[int]) -> list[tuple[int, list[Line]]]:
    by_page: dict[int, list[Line]] = defaultdict(list)
    for line in lines:
        by_page[line.pdf_page].append(line)
    return [(pdf, by_page.get(pdf, [])) for pdf in pages]


def _brief(page_ids: list[str]) -> str:
    shown = ", ".join(page_ids[:8])
    extra = f" (+{len(page_ids) - 8})" if len(page_ids) > 8 else ""
    return f"{shown}{extra}"
