"""A formula or table block written for a reader: its structure, or its passage.

The vault and the corpus differ only in the marker they write, so both call here with the
block's leading markers (low-confidence, orphaned, empty) and its body (the adopted text with an
open span's marker in place). A block that publishes structure keeps its leading markers above
it. A block with an open span publishes no structure: a formula is its passage beside the crop,
never math around a marker, and a table is its passage laid out as a grid when the passage still
reads as one.
"""

from llb.prep.book.grid import table_from_text
from llb.prep.book.render.mathform import (
    MARKDOWN,
    formula_math,
    render_formula,
    render_table,
    table_markdown,
    table_mode,
)
from llb.prep.book.render.types import RenderBlock


def has_open_span(block: RenderBlock) -> bool:
    """True while a span of this block is still unresolved."""
    return any(mark.reason == "unresolved" for mark in block.marks)


def formula_text(block: RenderBlock, lead: str, body: str, number: str, crop: str | None) -> str:
    """Math when the block publishes a formula, else its passage and the crop."""
    if block.formula is not None:
        math = render_formula(block.formula.latex, number, block.formula.display, crop)
        return _led(lead, math)
    if has_open_span(block):
        return _with_embed(_led_inline(lead, body), crop)
    if not body.strip():
        return _with_embed(lead, crop)
    return _with_embed(_led(lead, formula_math(body, number)), crop)


def table_text(
    block: RenderBlock, lead: str, body: str, image: str | None, evidence: str | None
) -> str:
    """The grid when the block publishes a table, else its passage, with the caption it prints."""
    if block.table is not None:
        emitted = render_table(block.table, image, evidence, emit_caption=bool(block.caption))
        return _led(lead, emitted)
    parts = [block.caption] if block.caption else []
    if body.strip():
        parts.append(passage_grid(body, open_span=has_open_span(block)))
    for embed in (image, evidence):
        if embed:
            parts.append(f"![[{embed}]]")
    return _led(lead, "\n\n".join(parts))


def passage_grid(body: str, *, open_span: bool = False) -> str:
    """A table passage as a Markdown grid when it parses as one, else the passage lines.

    A held-open passage that no longer reads as rows -- one marker for the whole table -- is
    written as it is rather than wrapped as a one-column table.
    """
    grid = table_from_text(body)
    if grid is not None and table_mode(grid) == MARKDOWN:
        return render_table(grid, None, None, emit_caption=False)
    if open_span and "|" not in body and "<table" not in body.lower():
        return body.strip()
    return table_markdown(body)


def _led(lead: str, body: str) -> str:
    if not lead:
        return body
    if not body:
        return lead
    return f"{lead}\n\n{body}"


def _led_inline(lead: str, body: str) -> str:
    return " ".join(piece for piece in (lead, body.strip()) if piece)


def _with_embed(text: str, embed: str | None) -> str:
    if not embed:
        return text
    if not text:
        return f"![[{embed}]]"
    return f"{text}\n\n![[{embed}]]"
