"""Locate printed blocks and vault anchors in the current corpus projection."""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.book_references import ReferenceTarget
from llb.goldset.markers import MARKER
from llb.prep.book.project.write import ProjectedDocument
from llb.prep.book.render.catalog import Target
from llb.prep.book.render.constants import CONTENT_DIRNAME


_PAGE_ANCHOR = re.compile(r"^p(\d+)$")


@dataclass(frozen=True, slots=True)
class ProjectedBlock:
    doc_id: str
    block_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class ProjectionLocations:
    by_anchor: dict[tuple[str, str], ProjectedBlock]
    by_page: dict[tuple[str, int], ProjectedBlock]
    by_id: dict[str, ProjectedBlock]
    text: dict[str, str]


def reference_target(block: ProjectedBlock) -> ReferenceTarget:
    return ReferenceTarget(
        doc_id=block.doc_id,
        block_id=block.block_id,
        char_start=block.char_start,
        char_end=block.char_end,
    )


def lookup_vault_target(
    target: Target | None, located: ProjectionLocations
) -> ProjectedBlock | None:
    if target is None:
        return None
    match = _PAGE_ANCHOR.match(target.anchor)
    if match is not None:
        return located.by_page.get((target.stem, int(match.group(1))))
    vault = f"{CONTENT_DIRNAME}/{target.stem}.md"
    return located.by_anchor.get((vault, target.anchor))


def locate_blocks(items: list[tuple[ProjectedDocument, str]]) -> ProjectionLocations:
    by_anchor: dict[tuple[str, str], ProjectedBlock] = {}
    by_page: dict[tuple[str, int], ProjectedBlock] = {}
    by_id: dict[str, ProjectedBlock] = {}
    text: dict[str, str] = {}
    for item, doc_id in items:
        text[doc_id] = item.text
        for page, raw, block in printed_blocks(item, doc_id):
            by_id.setdefault(block.block_id, block)
            anchor = raw.get("vault_anchor")
            vault = raw.get("vault_file")
            if isinstance(anchor, str) and isinstance(vault, str) and anchor:
                by_anchor.setdefault((vault, anchor), block)
            by_page.setdefault((item.stem, page), block)
    return ProjectionLocations(by_anchor, by_page, by_id, text)


def printed_blocks(
    item: ProjectedDocument, doc_id: str
) -> Iterator[tuple[int, JsonObject, ProjectedBlock]]:
    for page in item.pages:
        for raw in page.blocks:
            block = _block(doc_id, raw)
            if block is not None and _printed(item.text, block):
                yield page.page, raw, block


def _block(doc_id: str, raw: object) -> ProjectedBlock | None:
    if not isinstance(raw, dict) or raw.get("generated"):
        return None
    block_id = raw.get("block_id")
    start = raw.get("char_start")
    end = raw.get("char_end")
    if not isinstance(block_id, str) or not isinstance(start, int) or not isinstance(end, int):
        return None
    if end <= start or not block_id:
        return None
    return ProjectedBlock(doc_id, block_id, start, end)


def _printed(text: str, block: ProjectedBlock) -> bool:
    return bool(MARKER.sub("", text[block.char_start : block.char_end]).strip())
