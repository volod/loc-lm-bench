"""ASCII file names and the anchors a numbered target keeps when ordinals move.

A section, figure, table, or equation anchor is its printed number. A block id
contains the in-page ordinal, so it is only used to find the block, never to
name the anchor.
"""

import re

from llb.core.contracts.data_prep.book_structure import DocumentStructure, StructureNode
from llb.prep.book.render.constants import (
    FILE_INDEX_WIDTH,
    HEADING_LEVEL,
    ITEM_ANCHOR,
    LEVEL_ANCHOR,
    ORDINAL_WIDTH,
    SLUG_LIMIT,
)

_ASCII = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"[^a-z0-9]+")


def ascii_slug(text: str, limit: int = SLUG_LIMIT) -> str:
    """Lowercase ASCII words joined by hyphens. Empty when nothing survives."""
    return _ASCII.sub("-", text.lower()).strip("-")[:limit].strip("-")


def book_slug(pdf_name: str) -> str:
    """The vault directory name, from the PDF file name."""
    stem = pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name
    return ascii_slug(stem) or "book"


def source_filename(pdf_name: str) -> str:
    """The PDF's name inside ``source/``. An ASCII name is kept."""
    if re.fullmatch(r"[A-Za-z0-9._-]+", pdf_name) and pdf_name.lower().endswith(".pdf"):
        return pdf_name
    return f"{book_slug(pdf_name)}.pdf"


def node_label(node: StructureNode) -> str:
    """The heading a reader sees for one structure node."""
    if node.number and node.title:
        return f"{node.number} {node.title}"
    return node.title or node.number or node.level


def node_slug(node: StructureNode, book: str) -> str:
    """The file-name slug of one node. The book falls back to the vault slug."""
    if node.level == "book":
        title = ascii_slug(node.title)
        return book if not title or title == "book" else title
    parts: list[str] = [node.level]
    if node.number:
        parts.append(node.number)
    if node.title:
        parts.append(node.title)
    return ascii_slug("-".join(parts)) or node.level


def content_stem(
    index: int,
    node: StructureNode,
    book: str,
    pdf_start: int,
    pdf_end: int,
    part_no: int,
    part_count: int,
    span_count: int,
) -> str:
    """``0003_section-1-1-purpose`` or, once a section is split, a page-range suffix."""
    stem = f"{index:0{FILE_INDEX_WIDTH}d}_{node_slug(node, book)}"
    if part_count > 1 or span_count > 1:
        stem += f"_p{pdf_start:04d}-p{pdf_end:04d}-{part_no}"
    return stem


def figure_asset(chapter: str, page_id: str, ordinal: int, kind: str, number: str) -> str:
    """``assets/<chapter>/p0007_0003_figure`` plus a number slug when one is known."""
    stem = f"{page_id}_{ordinal:0{ORDINAL_WIDTH}d}_{ascii_slug(kind) or 'figure'}"
    slug = ascii_slug(number)
    if slug:
        stem = f"{stem}_{slug}"
    return f"assets/{ascii_slug(chapter) or 'book'}/{stem}.png"


def anchor_map(structure: DocumentStructure) -> dict[str, str]:
    """Block id to anchor. Headings win when a block is also a numbered item."""
    taken: set[str] = set()
    found: dict[str, str] = {}
    for node in walk(structure.hierarchy):
        if node.block_id and node.number and node.level in LEVEL_ANCHOR:
            _claim(found, taken, node.block_id, _level(node.level, node.number), node.pdf_start)
    for item in structure.inventory:
        prefix = ITEM_ANCHOR.get(item.kind)
        if prefix is None:
            continue
        _claim(found, taken, item.block_id, _numbered(prefix, item.number), item.pdf_page)
    return found


def equation_numbers(structure: DocumentStructure) -> dict[str, str]:
    """Block id to the printed equation number the inventory recorded."""
    return {item.block_id: item.number for item in structure.inventory if item.kind == "equation"}


def figure_numbers(structure: DocumentStructure) -> dict[str, str]:
    """Block id to the printed figure number, for the asset slug."""
    return {item.block_id: item.number for item in structure.inventory if item.kind == "figure"}


def heading_levels(structure: DocumentStructure) -> dict[str, int]:
    """Block id of a heading node to its Markdown level."""
    levels: dict[str, int] = {}
    for node in walk(structure.hierarchy):
        if node.block_id and node.level in HEADING_LEVEL:
            levels[node.block_id] = HEADING_LEVEL[node.level]
    return levels


def structure_roles(structure: DocumentStructure) -> dict[str, str]:
    """Block id to the role structure already earned: heading, header, page number."""
    roles: dict[str, str] = {}
    for page in structure.pages:
        if page.block_id:
            roles[page.block_id] = "page-number"
    for header in structure.headers:
        for block_id in header.block_ids:
            roles.setdefault(block_id, "running-header")
    for node in walk(structure.hierarchy):
        if node.block_id and node.level != "book":
            roles[node.block_id] = "heading"
    return roles


def walk(node: StructureNode) -> list[StructureNode]:
    """The node and its descendants, preorder."""
    found = [node]
    for child in node.children:
        found.extend(walk(child))
    return found


def parents_of(root: StructureNode) -> dict[str, StructureNode | None]:
    """Node id to its parent. The book has none."""
    found: dict[str, StructureNode | None] = {root.node_id: None}

    def visit(node: StructureNode) -> None:
        for child in node.children:
            found[child.node_id] = node
            visit(child)

    visit(root)
    return found


def chapter_slug(node: StructureNode, parents: dict[str, StructureNode | None], book: str) -> str:
    """The chapter directory a figure in this node is filed under."""
    current: StructureNode | None = node
    while current is not None:
        if current.level == "chapter":
            return node_slug(current, book)
        current = parents.get(current.node_id)
    return node_slug(node, book)


def _level(level: str, number: str) -> str:
    return _numbered(LEVEL_ANCHOR[level], number)


def _numbered(prefix: str, number: str) -> str:
    slug = _NUMBER.sub("-", number.lower()).strip("-")
    return f"{prefix}-{slug}" if slug else prefix


def _claim(found: dict[str, str], taken: set[str], block_id: str, base: str, pdf_page: int) -> None:
    if block_id in found:
        return
    name = base if base not in taken else f"{base}-p{pdf_page:04d}"
    suffix = 2
    while name in taken:
        name = f"{base}-p{pdf_page:04d}-{suffix}"
        suffix += 1
    taken.add(name)
    found[block_id] = name
