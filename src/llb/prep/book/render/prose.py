"""Markdown for one resolved block: the text a reader sees, and nothing raw.

Formulas become display math with the printed equation number. Tables become
Markdown or HTML. Figures become embeds. Furniture stays visible and labelled.
"""

from collections.abc import Callable

from llb.prep.book.render.mathform import formula_math, render_formula, render_table
from llb.prep.book.render.types import RenderBlock, RenderContext, Unit
from llb.prep.book.review.markers import render_marker, render_passage

Href = Callable[[int], str]


def passage(block: RenderBlock, href: Href) -> str:
    """The block's resolved text, with a marker where a span was left unresolved."""
    if block.resolved is None:
        return block.text
    link = href(block.pdf_page)
    crops = {(start, end): name for start, end, name in block.crops}
    replacements: dict[tuple[int, int], str] = {}
    prefixes: list[str] = []
    for mark in block.marks:
        marker = render_marker(
            mark.kind, link, mark.block_id, mark.start, mark.end, crops.get((mark.start, mark.end))
        )
        if mark.reason == "unresolved":
            replacements[(mark.start, mark.end)] = marker
        else:
            prefixes.append(marker)
    return render_passage(block.resolved, replacements, prefixes)


def join_lines(blocks: tuple[RenderBlock, ...] | list[RenderBlock], href: Href) -> str:
    """Consecutive body lines of one paragraph, dehyphenated at a line break."""
    text = ""
    for block in blocks:
        piece = passage(block, href).strip()
        text = _append(text, piece)
    return text


def render_units(units: list[Unit], ctx: RenderContext, seen_pages: set[int]) -> str:
    """The body of one file. The first block of a PDF page carries that page's anchor."""
    chunks: list[str] = []
    for unit in units:
        anchor = _page_anchor(unit, seen_pages)
        if anchor:
            chunks.append(anchor)
        body = render_unit(unit, ctx)
        if body:
            chunks.append(body)
    return "\n\n".join(chunks)


def render_unit(unit: Unit, ctx: RenderContext) -> str:
    """One unit, then the numbered-target anchors that belong to its blocks."""
    body = _BODY.get(unit.role, _paragraph)(unit, ctx)
    return _with_anchors(body, unit, ctx)


def table_markdown(text: str) -> str:
    """HTML when the text is already a table, Markdown when it has pipes, else one column."""
    stripped = text.strip()
    lowered = stripped.lower()
    if "<table" in lowered and "</table>" in lowered:
        return stripped
    rows = [row.strip() for row in stripped.splitlines() if row.strip()]
    if rows and all("|" in row for row in rows):
        return "\n".join(rows)
    body = ["| |", "| --- |"]
    for row in rows or [stripped]:
        body.append("| " + row.replace("|", "\\|") + " |")
    return "\n".join(body)


def _append(text: str, piece: str) -> str:
    if not piece:
        return text
    if not text:
        return piece
    if text.endswith("-") and piece[0].islower():
        return text[:-1] + piece
    return f"{text} {piece}"


def _page_anchor(unit: Unit, seen: set[int]) -> str:
    for block in unit.blocks:
        if block.pdf_page in seen:
            continue
        seen.add(block.pdf_page)
        return f"^{block.page_id}"
    return ""


def _with_anchors(body: str, unit: Unit, ctx: RenderContext) -> str:
    if not unit.keep_anchors:
        return body
    names: list[str] = []
    for block in unit.blocks:
        name = ctx.anchors.get(block.block_id, "")
        if name and name not in names:
            names.append(name)
    if not names:
        return body
    tail = "\n\n".join(f"^{name}" for name in names)
    return f"{body}\n\n{tail}" if body else tail


def _paragraph(unit: Unit, ctx: RenderContext) -> str:
    if _suppressed(unit, ctx):
        return ""
    return unit.text.strip()


def _suppressed(unit: Unit, ctx: RenderContext) -> bool:
    return bool(unit.blocks) and all(block.block_id in ctx.suppressed for block in unit.blocks)


def _formula(unit: Unit, ctx: RenderContext) -> str:
    block = unit.blocks[0]
    number = ""
    for item in unit.blocks:
        number = ctx.equations.get(item.block_id, number)
    if block.formula is not None and block.formula.equation_number:
        number = block.formula.equation_number
    if block.formula is not None:
        return render_formula(
            block.formula.latex,
            number,
            block.formula.display,
            ctx.formula_crops.get(block.block_id),
        )
    body = formula_math(unit.text, number)
    crop = ctx.formula_crops.get(block.block_id)
    if not crop:
        return body
    return f"{body}\n\n![[{crop}]]"


def _table(unit: Unit, ctx: RenderContext) -> str:
    block = unit.blocks[0]
    if block.table is None:
        return table_markdown(unit.text)
    return render_table(
        block.table,
        ctx.table_images.get(block.block_id),
        ctx.table_evidence.get(block.block_id),
        emit_caption=block.emit_caption,
    )


def _heading(unit: Unit, ctx: RenderContext) -> str:
    block = unit.blocks[0]
    level = ctx.levels.get(block.block_id, 2)
    return f"{'#' * level} {unit.text.strip()}"


def _furniture(unit: Unit, _ctx: RenderContext) -> str:
    text = unit.text.strip()
    return f"> [{unit.role}] {text}".rstrip()


def _figure(unit: Unit, ctx: RenderContext) -> str:
    return "\n\n".join(
        line for line in (_figure_block(block, ctx) for block in unit.blocks) if line
    )


def _pair(unit: Unit, ctx: RenderContext) -> str:
    lines: list[str] = []
    for block in unit.blocks:
        if block.block_id in ctx.suppressed:
            continue
        if block.kind == "figure" or block.role == "figure":
            lines.append(_figure_block(block, ctx))
        else:
            lines.append(passage(block, ctx.content_href).strip())
    return "\n\n".join(line for line in lines if line)


def _figure_block(block: RenderBlock, ctx: RenderContext) -> str:
    if block.block_id in ctx.suppressed:
        return ""
    lines: list[str] = []
    path = ctx.assets.get(block.block_id, "")
    if path:
        lines.append(f"![[{path}]]")
    piece = passage(block, ctx.content_href).strip()
    if piece:
        lines.append(piece)
    caption = ctx.figure_captions.get(block.block_id, "").strip()
    if caption and caption != piece:
        lines.append(caption)
    if path:
        lines.append(f"[{block.page_id}]({ctx.content_href(block.pdf_page)})")
    if lines:
        return "\n\n".join(lines)
    href = ctx.content_href(block.pdf_page)
    return f'[!figure-missing]({href} "{block.page_id} {block.block_id}")'


def _blank(unit: Unit, ctx: RenderContext) -> str:
    block = unit.blocks[0]
    href = ctx.content_href(block.pdf_page)
    return f'[!blank]({href} "{block.page_id}")'


def _omission(unit: Unit, ctx: RenderContext) -> str:
    lines: list[str] = []
    for block in unit.blocks:
        href = ctx.content_href(block.pdf_page)
        crop = block.crops[0][2] if block.crops else None
        lines.append(render_marker("unreadable", href, block.block_id, 0, 0, crop))
    return "\n\n".join(lines)


def _failed(unit: Unit, ctx: RenderContext) -> str:
    block = unit.blocks[0]
    href = ctx.content_href(block.pdf_page)
    reason = block.fail_reason or "recovery failed"
    marker = f'[!page-failed]({href} "{block.page_id} {reason}")'
    if not block.crops:
        return marker
    return f"{marker}\n\n![[{block.crops[0][2]}]]"


def _missing(unit: Unit, ctx: RenderContext) -> str:
    block = unit.blocks[0]
    href = ctx.content_href(block.pdf_page)
    return f'[!unreadable]({href} "{block.block_id} 0-0")'


_BODY = {
    "paragraph": _paragraph,
    "formula": _formula,
    "table": _table,
    "heading": _heading,
    "figure": _figure,
    "pair": _pair,
    "caption": _paragraph,
    "footnote": _paragraph,
    "unknown": _paragraph,
    "page-failed": _failed,
    "missing-block": _missing,
    "blank": _blank,
    "omission": _omission,
    "running-header": _furniture,
    "page-number": _furniture,
    "ornament": _furniture,
    "neighbour-page-bleed": _furniture,
}
