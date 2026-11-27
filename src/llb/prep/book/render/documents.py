"""Content notes, the table of contents, and the navigation that joins them.

Every content note opens and closes with the same links: contents, the
neighbouring notes, and the PDF page the note starts on. Front matter is only
the fields a vault or a later retrieval pass reads.
"""

import json

from llb.core.contracts.data_prep.book_structure import DocumentStructure, StructureNode
from llb.prep.book.render.constants import (
    CONTENT_DIRNAME,
    HEADING_LEVEL,
    LIST_FILENAMES,
    TOC_FILENAME,
)
from llb.prep.book.render.grammar import ReferenceGrammar
from llb.prep.book.render.lists import render_lists
from llb.prep.book.render.references import ReferenceCounts, apply_references
from llb.prep.book.render.names import content_stem, node_label
from llb.prep.book.render.prose import render_units
from llb.prep.book.render.segment import Span, pack_span
from llb.prep.book.render.types import RenderContext, Unit, VaultFile
from llb.prep.book.review.markers import parse_markers


def layout_files(spans: list[Span], cap: int, book: str, ctx: RenderContext) -> list[VaultFile]:
    """One note per section part, in reading order, with its body already rendered."""
    packed = [(span, pack_span(span, cap, ctx.content_href)) for span in spans]
    span_count: dict[str, int] = {}
    for span, _parts in packed:
        span_count[span.node.node_id] = span_count.get(span.node.node_id, 0) + 1
    files: list[VaultFile] = []
    used: set[str] = set()
    seen: set[int] = set()
    opened: set[str] = set()
    index = 1
    for span, parts in packed:
        opening = span.node.node_id not in opened
        for part_no, units in enumerate(parts, start=1):
            files.append(
                _file(
                    index,
                    span,
                    units,
                    part_no,
                    len(parts),
                    span_count,
                    book,
                    ctx,
                    used,
                    seen,
                    opening,
                )
            )
            index += 1
        opened.add(span.node.node_id)
    return files


def render_markdown(
    files: list[VaultFile],
    structure: DocumentStructure,
    ctx: RenderContext,
    grammar: ReferenceGrammar,
) -> tuple[dict[str, str], ReferenceCounts]:
    """Relative path to Markdown, including the contents and the generated lists."""
    counts, catalog = apply_references(files, structure, grammar)
    docs = {TOC_FILENAME: _toc(files, structure, ctx)}
    docs.update(render_lists(catalog))
    for index, file in enumerate(files):
        docs[f"{CONTENT_DIRNAME}/{file.stem}.md"] = _wrap(index, files, structure, ctx)
    return docs, counts


def front_matter(
    book: str,
    node_id: str,
    pdf_start: int,
    pdf_end: int,
    printed_start: str | None,
    printed_end: str | None,
    review_count: int,
) -> str:
    """YAML. Printed pages stay strings so a roman numeral is not read as a number."""
    rows = [
        "---",
        f"book: {json.dumps(book)}",
        f"node_id: {json.dumps(node_id)}",
        f"pdf_start: {pdf_start}",
        f"pdf_end: {pdf_end}",
        f"printed_start: {_yaml(printed_start)}",
        f"printed_end: {_yaml(printed_end)}",
        f"review_count: {review_count}",
        "---",
    ]
    return "\n".join(rows)


def _file(
    index: int,
    span: Span,
    units: list[Unit],
    part_no: int,
    part_count: int,
    span_count: dict[str, int],
    book: str,
    ctx: RenderContext,
    used: set[str],
    seen: set[int],
    opening: bool,
) -> VaultFile:
    start, end = _range(units, span.node)
    stem = content_stem(
        index,
        span.node,
        book,
        start,
        end,
        part_no,
        part_count,
        span_count[span.node.node_id],
    )
    stem = _unique(stem, used)
    page_anchors = _local_page_anchors(units, seen, index)
    body = render_units(units, ctx, page_anchors)
    if opening and not any(unit.role == "heading" for unit in units):
        body = _title(span.node, body)
    return VaultFile(
        stem, span.node, units, start, end, span.depth, body, _reviews(body), opening, page_anchors
    )


def _local_page_anchors(units: list[Unit], seen: set[int], note_index: int) -> dict[int, str]:
    anchors: dict[int, str] = {}
    for unit in units:
        for block in unit.blocks:
            page = block.pdf_page
            if page not in anchors:
                label = block.page_id
                anchors[page] = f"{label}-note-{note_index}" if page in seen else label
                seen.add(page)
    return anchors


def _wrap(
    index: int, files: list[VaultFile], structure: DocumentStructure, ctx: RenderContext
) -> str:
    file = files[index]
    printed_start, printed_end = _printed(structure, file.pdf_start, file.pdf_end)
    matter = front_matter(
        ctx.book,
        file.node.node_id,
        file.pdf_start,
        file.pdf_end,
        printed_start,
        printed_end,
        file.review_count,
    )
    nav = _nav(index, files, structure, ctx)
    return f"{matter}\n\n{nav}\n\n{file.body}\n\n{nav}\n"


def _nav(
    index: int, files: list[VaultFile], structure: DocumentStructure, ctx: RenderContext
) -> str:
    file = files[index]
    bits = [f"[Contents](../{TOC_FILENAME})"]
    if index > 0:
        bits.append(f"[Previous]({files[index - 1].stem}.md)")
    if index + 1 < len(files):
        bits.append(f"[Next]({files[index + 1].stem}.md)")
    label = f"PDF {file.pdf_start}-{file.pdf_end}"
    label += _printed_label(*_printed(structure, file.pdf_start, file.pdf_end))
    bits.append(f"[{label}]({ctx.content_href(file.pdf_start)})")
    if ctx.fallback:
        bits.extend(_page_links(file, ctx))
    return " | ".join(bits)


def _toc(files: list[VaultFile], structure: DocumentStructure, ctx: RenderContext) -> str:
    start = structure.pages[0].pdf_page
    end = structure.pages[-1].pdf_page
    printed_start, printed_end = _printed(structure, start, end)
    matter = front_matter(ctx.book, "book", start, end, printed_start, printed_end, 0)
    nav = _toc_nav(files, ctx, start, end, structure)
    lines = _toc_lines(structure.hierarchy, _by_node(files))
    hierarchy = "\n".join(lines) if lines else "No sections were rendered."
    body = "# Contents\n\n" + "\n".join(_list_links()) + "\n\n" + hierarchy
    return f"{matter}\n\n{nav}\n\n{body}\n\n{nav}\n"


def _toc_nav(
    files: list[VaultFile],
    ctx: RenderContext,
    start: int,
    end: int,
    structure: DocumentStructure,
) -> str:
    bits: list[str] = []
    if files:
        bits.append(f"[Next]({CONTENT_DIRNAME}/{files[0].stem}.md)")
    label = f"PDF {start}-{end}" + _printed_label(*_printed(structure, start, end))
    bits.append(f"[{label}]({ctx.toc_href(start)})")
    return " | ".join(bits)


def _toc_lines(node: StructureNode, by_node: dict[str, list[VaultFile]]) -> list[str]:
    lines: list[str] = []
    if node.level == "book":
        lines.extend(_book_items(by_node.get(node.node_id, [])))
        child_depth = 0
    else:
        lines.extend(_item(node, by_node.get(node.node_id, []), 0))
        child_depth = 1
    for child in node.children:
        lines.extend(_child_lines(child, by_node, child_depth))
    return lines


def _child_lines(node: StructureNode, by_node: dict[str, list[VaultFile]], depth: int) -> list[str]:
    lines = _item(node, by_node.get(node.node_id, []), depth)
    for child in node.children:
        lines.extend(_child_lines(child, by_node, depth + 1))
    return lines


def _list_links() -> list[str]:
    return [f"- [{title}]({filename})" for _kind, filename, title in LIST_FILENAMES]


def _book_items(files: list[VaultFile]) -> list[str]:
    shown = [file for file in files if file.opening] or files
    return [f"- [{_note_label(file)}]({CONTENT_DIRNAME}/{file.stem}.md)" for file in shown]


def _item(node: StructureNode, files: list[VaultFile], depth: int) -> list[str]:
    pad = "  " * depth
    label = node_label(node)
    shown = [file for file in files if file.opening]
    if not shown:
        return [f"{pad}- {label}"]
    lines = [f"{pad}- [{label}]({CONTENT_DIRNAME}/{shown[0].stem}.md)"]
    for extra in shown[1:]:
        lines.append(
            f"{pad}  - [PDF {extra.pdf_start}-{extra.pdf_end}]({CONTENT_DIRNAME}/{extra.stem}.md)"
        )
    return lines


def _note_label(file: VaultFile) -> str:
    if file.node.level == "book":
        return f"PDF {file.pdf_start}-{file.pdf_end}"
    return node_label(file.node)


def _title(node: StructureNode, body: str) -> str:
    level = HEADING_LEVEL.get(node.level, 2)
    heading = f"{'#' * level} {node_label(node)}"
    return f"{heading}\n\n{body}" if body else heading


def _page_links(file: VaultFile, ctx: RenderContext) -> list[str]:
    links: list[str] = []
    seen: list[str] = []
    for unit in file.units:
        for block in unit.blocks:
            if block.page_id in seen or block.pdf_page not in ctx.pages_with_images:
                continue
            seen.append(block.page_id)
            links.append(f"[{block.page_id}](../source/pages/{block.page_id}.png)")
    return links


def _printed(structure: DocumentStructure, start: int, end: int) -> tuple[str | None, str | None]:
    by_page = {page.pdf_page: page.printed for page in structure.pages}
    return by_page.get(start), by_page.get(end)


def _printed_label(start: str | None, end: str | None) -> str:
    if start and end and start != end:
        return f", printed {start}-{end}"
    if start or end:
        return f", printed {start or end}"
    return ""


def _reviews(body: str) -> int:
    return len(parse_markers(body)) + body.count("[!page-failed]") + body.count("[!figure-missing]")


def _range(units: list[Unit], node: StructureNode) -> tuple[int, int]:
    pages = [block.pdf_page for unit in units for block in unit.blocks]
    if not pages:
        return node.pdf_start, node.pdf_end
    return min(pages), max(pages)


def _by_node(files: list[VaultFile]) -> dict[str, list[VaultFile]]:
    grouped: dict[str, list[VaultFile]] = {}
    for file in files:
        grouped.setdefault(file.node.node_id, []).append(file)
    return grouped


def _unique(stem: str, used: set[str]) -> str:
    name = stem
    suffix = 2
    while name in used:
        name = f"{stem}-{suffix}"
        suffix += 1
    used.add(name)
    return name


def _yaml(value: str | None) -> str:
    if value is None:
        return "null"
    return json.dumps(value)
