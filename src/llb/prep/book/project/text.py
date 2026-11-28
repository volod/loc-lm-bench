"""Project one vault content file into corpus text and page citations.

The text is the resolved reading: formulas as LaTeX, tables as rendered
Markdown, and an unresolved span as a machine marker. Navigation, front
matter, and page furniture stay out. Character offsets in the citation
sidecar are indexes into this text.
"""

from collections.abc import Callable
from dataclasses import replace

from llb.core.contracts.data_prep.book_blocks import Region
from llb.goldset.markers import MARKER, marker_token
from llb.prep.book.project.spans import Fragment, RenderedDocument, assemble
from llb.prep.book.render.constants import CONTENT_DIRNAME, FURNITURE_ROLES, HEADING_LEVEL
from llb.prep.book.render.names import node_label
from llb.prep.book.render.structured import formula_text, table_text
from llb.prep.book.render.types import RenderBlock, RenderContext, Unit, VaultFile
from llb.prep.book.review.markers import render_passage

Render = Callable[[Unit, RenderContext, str], list[Fragment]]


def render_content(file: VaultFile, ctx: RenderContext) -> RenderedDocument | None:
    """The corpus document for one content file, or none when it has no content."""
    vault_file = f"{CONTENT_DIRNAME}/{file.stem}.md"
    fragments = _fragments(file, ctx, vault_file)
    if not fragments:
        return None
    anchors = {line[1:] for line in file.body.splitlines() if line.startswith("^")}
    document = assemble(
        [
            fragment
            if fragment.anchor in anchors
            else replace(fragment, anchor=file.page_anchors[fragment.page])
            for fragment in fragments
        ]
    )
    if not document.text.strip():
        return None
    return document


def corpus_passage(block: RenderBlock) -> str:
    """The block a retrieval document keeps. An unresolved span is a marker."""
    standing = _standing_token(block)
    if standing is not None or block.resolved is None:
        return standing or ""
    replacements, prefixes = _tokens(block)
    return render_passage(block.resolved, replacements, prefixes)


def corpus_parts(block: RenderBlock) -> tuple[str, str]:
    """The block's leading tokens, and its body with an open span's token in place."""
    standing = _standing_token(block)
    if standing is not None or block.resolved is None:
        return "", standing or ""
    replacements, prefixes = _tokens(block)
    return " ".join(prefixes), render_passage(block.resolved, replacements, ())


def _standing_token(block: RenderBlock) -> str | None:
    """The one token a blank, unread, failed, or missing block stands as."""
    if block.role == "blank":
        return marker_token("blank", block.page_id or "page", 0, 0)
    if block.role == "omission":
        return marker_token("omission", block.block_id or block.page_id or "page", 0, 0)
    if block.failed or block.role == "page-failed":
        return marker_token("page-failed", block.page_id or "page", 0, 0)
    if block.resolved is None:
        label = block.block_id or block.page_id or "page"
        return marker_token("unresolved", label, 0, 0)
    return None


def _tokens(block: RenderBlock) -> tuple[dict[tuple[int, int], str], list[str]]:
    replacements: dict[tuple[int, int], str] = {}
    prefixes: list[str] = []
    for mark in block.marks:
        token = marker_token(mark.reason, mark.block_id, mark.start, mark.end)
        if mark.reason == "unresolved":
            replacements[(mark.start, mark.end)] = token
        else:
            prefixes.append(token)
    return replacements, prefixes


def _fragments(file: VaultFile, ctx: RenderContext, vault_file: str) -> list[Fragment]:
    out: list[Fragment] = []
    if file.opening and not any(unit.role == "heading" for unit in file.units):
        _extend(out, [_title(file, ctx, vault_file)])
    for unit in file.units:
        if unit.role in FURNITURE_ROLES:
            continue
        _extend(out, _render_unit(unit, ctx, vault_file))
    return out


def _extend(out: list[Fragment], batch: list[Fragment]) -> None:
    if not batch:
        return
    if out and not batch[0].gap_before:
        batch = [replace(batch[0], gap_before="\n\n"), *batch[1:]]
    out.extend(batch)


def _render_unit(unit: Unit, ctx: RenderContext, vault_file: str) -> list[Fragment]:
    renderer = _UNIT.get(unit.role, _plain)
    return renderer(unit, ctx, vault_file)


def _title(file: VaultFile, ctx: RenderContext, vault_file: str) -> Fragment:
    level = HEADING_LEVEL.get(file.node.level, 1)
    text = f"{'#' * level} {node_label(file.node)}"
    block_id = file.node.block_id or ""
    anchor = ctx.anchors.get(block_id, "") or f"p{file.pdf_start:04d}"
    return Fragment(file.pdf_start, text, block_id, None, anchor, vault_file, "", generated=True)


def _paragraph(unit: Unit, ctx: RenderContext, vault_file: str) -> list[Fragment]:
    pieces: list[Fragment] = []
    for block in unit.blocks:
        text = corpus_passage(block).strip()
        if not text:
            continue
        gap = ""
        if pieces and not _dehyphenate(pieces, text):
            gap = " "
        pieces.append(_fragment(block, text, ctx, vault_file, gap))
    if unit.text_range is None:
        return pieces
    return _clip(pieces, *unit.text_range)


def _clip(pieces: list[Fragment], start: int, end: int) -> list[Fragment]:
    """Project only the part of a plain paragraph that this note renders."""
    clipped: list[Fragment] = []
    cursor = 0
    for fragment in pieces:
        text_start = cursor + len(fragment.gap_before)
        stop = text_start + len(fragment.text)
        if text_start < end and stop > start:
            text = fragment.text[max(start - text_start, 0) : end - text_start]
            gap = fragment.gap_before if clipped else ""
            clipped.append(replace(fragment, text=text, gap_before=gap))
        cursor = stop
    return clipped


def _dehyphenate(pieces: list[Fragment], text: str) -> bool:
    previous = pieces[-1].text
    if not previous.endswith("-") or not text[:1].islower():
        return False
    pieces[-1] = replace(pieces[-1], text=previous[:-1])
    return True


def _pair(unit: Unit, ctx: RenderContext, vault_file: str) -> list[Fragment]:
    pieces: list[Fragment] = []
    for block in unit.blocks:
        text = _styled(block, ctx, corpus_passage(block))
        if not text:
            continue
        gap = "\n\n" if pieces else ""
        pieces.append(_fragment(block, text, ctx, vault_file, gap))
    return pieces


def _one(style: Callable[[Unit, RenderContext, str], str]) -> Render:
    def render(unit: Unit, ctx: RenderContext, vault_file: str) -> list[Fragment]:
        block = unit.blocks[0]
        text = style(unit, ctx, corpus_passage(block)).strip()
        if not text:
            return []
        return [_fragment(block, text, ctx, vault_file, "")]

    return render


def _plain(unit: Unit, ctx: RenderContext, vault_file: str) -> list[Fragment]:
    return _one(lambda _unit, _ctx, passage: passage)(unit, ctx, vault_file)


def _formula_text(unit: Unit, ctx: RenderContext, passage: str) -> str:
    stripped = passage.strip()
    if _marker_only(stripped):
        return stripped
    block = unit.blocks[0]
    number = ""
    for item in unit.blocks:
        number = ctx.equations.get(item.block_id, number)
    if block.formula is not None and block.formula.equation_number:
        number = block.formula.equation_number
    lead, body = corpus_parts(block)
    return formula_text(block, lead, body, number, None)


def _table_text(unit: Unit, _ctx: RenderContext, passage: str) -> str:
    stripped = passage.strip()
    if _marker_only(stripped):
        return stripped
    lead, body = corpus_parts(unit.blocks[0])
    return table_text(unit.blocks[0], lead, body, None, None)


def _heading_text(unit: Unit, ctx: RenderContext, passage: str) -> str:
    stripped = passage.strip()
    if _marker_only(stripped):
        return stripped
    level = ctx.levels.get(unit.blocks[0].block_id, 2)
    return f"{'#' * level} {stripped}"


def _styled(block: RenderBlock, ctx: RenderContext, passage: str) -> str:
    if block.role == "formula" or block.kind == "formula":
        return _formula_text_block(block, ctx, passage)
    if block.role == "table" or block.kind == "table":
        return _table_text(Unit("table", (block,), passage, False), ctx, passage)
    return passage.strip()


def _formula_text_block(block: RenderBlock, ctx: RenderContext, passage: str) -> str:
    unit = Unit("formula", (block,), passage, False)
    return _formula_text(unit, ctx, passage)


def _marker_only(text: str) -> bool:
    return MARKER.fullmatch(text) is not None


def _fragment(
    block: RenderBlock, text: str, ctx: RenderContext, vault_file: str, gap: str
) -> Fragment:
    return Fragment(
        page=block.pdf_page,
        text=text,
        block_id=block.block_id or block.page_id,
        region=_region(block.region),
        anchor=ctx.anchors.get(block.block_id, "") or block.page_id,
        vault_file=vault_file,
        gap_before=gap,
    )


def _region(region: Region | None) -> list[float] | None:
    if region is None:
        return None
    return [region.x0, region.y0, region.x1, region.y1]


_UNIT: dict[str, Render] = {
    "paragraph": _paragraph,
    "pair": _pair,
    "formula": _one(_formula_text),
    "table": _one(_table_text),
    "heading": _one(_heading_text),
}
