"""Markdown for one resolved block: the text a reader sees, and nothing raw.

Formulas become display math with the printed equation number. Tables become
Markdown or HTML. Figures become embeds. Furniture stays visible and labelled.
"""

from collections.abc import Callable

from llb.prep.book.render.structured import formula_text, table_text
from llb.prep.book.render.types import RenderBlock, RenderContext, Unit
from llb.prep.book.review.markers import render_marker, render_passage

Href = Callable[[int], str]


def passage(block: RenderBlock, href: Href) -> str:
    """The block's resolved text, with a marker where a span was left unresolved."""
    if block.resolved is None:
        return block.text
    replacements, prefixes = _markers(block, href)
    return render_passage(block.resolved, replacements, prefixes)


def passage_parts(block: RenderBlock, href: Href) -> tuple[str, str]:
    """The block's leading markers, and its body with an open span's marker in place."""
    if block.resolved is None:
        return "", block.text
    replacements, prefixes = _markers(block, href)
    return " ".join(prefixes), render_passage(block.resolved, replacements, ())


def _markers(block: RenderBlock, href: Href) -> tuple[dict[tuple[int, int], str], list[str]]:
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
    return replacements, prefixes


def join_lines(blocks: tuple[RenderBlock, ...] | list[RenderBlock], href: Href) -> str:
    """Consecutive body lines of one paragraph, dehyphenated at a line break."""
    text = ""
    for block in blocks:
        piece = passage(block, href).strip()
        text = _append(text, piece)
    return text


def render_units(units: list[Unit], ctx: RenderContext, page_anchors: dict[int, str]) -> str:
    """Render one note with a local anchor at the first unit from each of its pages."""
    chunks: list[str] = []
    seen: set[int] = set()
    for unit in units:
        for block in unit.blocks:
            if block.pdf_page not in seen:
                seen.add(block.pdf_page)
                chunks.append(f"^{page_anchors[block.pdf_page]}")
        body = render_unit(unit, ctx)
        if body:
            chunks.append(body)
    return "\n\n".join(chunks)


def render_unit(unit: Unit, ctx: RenderContext) -> str:
    """One unit, then the numbered-target anchors that belong to its blocks."""
    body = _BODY.get(unit.role, _paragraph)(unit, ctx)
    return _with_anchors(body, unit, ctx)


def _append(text: str, piece: str) -> str:
    if not piece:
        return text
    if not text:
        return piece
    if text.endswith("-") and piece[0].islower():
        return text[:-1] + piece
    return f"{text} {piece}"


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
    lead, body = passage_parts(block, ctx.content_href)
    return formula_text(block, lead, body, number, ctx.formula_crops.get(block.block_id))


def _table(unit: Unit, ctx: RenderContext) -> str:
    block = unit.blocks[0]
    lead, body = passage_parts(block, ctx.content_href)
    return table_text(
        block,
        lead,
        body,
        ctx.table_images.get(block.block_id),
        ctx.table_evidence.get(block.block_id),
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
    caption_block = ctx.figure_captions.get(block.block_id)
    caption = "" if caption_block is None else passage(caption_block, ctx.content_href).strip()
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
