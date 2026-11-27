"""Whether a formula or a table keeps its page crop.

A formula is below the validity bar when the Obsidian parser rejects it, and
below the agreement bar when engines disagree or the reading's confidence is
under the review cut. A table keeps its image when HTML cannot hold the grid,
and a review crop when a cell dispute names a dangerous substitution.
"""

from llb.core.contracts.data_prep.book_blocks import Block, RawReading
from llb.prep.book.metrics.latex_norm import canonicalize
from llb.prep.book.render.mathform import IMAGE, render_body, table_mode
from llb.prep.book.review.constants import LOW_CONFIDENCE_BELOW


def formula_needs_crop(block: Block, reading: RawReading | None) -> bool:
    """True when the formula is below the validity bar or the agreement bar."""
    if block.kind != "formula":
        return False
    latex = _latex(reading)
    if render_body(latex) is None:
        return True
    if reading is not None and reading.confidence is not None:
        if reading.confidence < LOW_CONFIDENCE_BELOW:
            return True
    if block.uncertainty:
        return True
    return _formulas_disagree(block)


def table_needs_image(block: Block, reading: RawReading | None) -> bool:
    """True when the native grid is not a faithful HTML table."""
    if block.kind != "table" or reading is None or reading.table is None:
        return False
    return table_mode(reading.table) == IMAGE


def table_needs_evidence(block: Block) -> bool:
    """True when a cell dispute should keep the source pixels beside the grid."""
    return block.kind == "table" and bool(block.cell_disputes)


def caption_is_separate(caption: str, captions: set[str], own: str) -> bool:
    """True when the table has no caption of its own, or some other block already prints it."""
    if not caption:
        return True
    others = set(captions)
    others.discard(" ".join(own.split()))
    return " ".join(caption.split()) in others


def _latex(reading: RawReading | None) -> str:
    if reading is None:
        return ""
    if reading.formula is not None:
        return reading.formula.latex
    return reading.text


def _formulas_disagree(block: Block) -> bool:
    bodies: list[str] = []
    numbers: list[str] = []
    for reading in block.readings:
        if reading.formula is None:
            continue
        try:
            bodies.append(canonicalize(reading.formula.latex))
        except ValueError:
            bodies.append(reading.formula.latex)
        numbers.append(reading.formula.equation_number or "")
    if len(set(bodies)) > 1:
        return True
    return len(set(numbers)) > 1
