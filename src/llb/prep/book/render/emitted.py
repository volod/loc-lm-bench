"""Which formula or table structure a resolved block publishes.

An engine's structure is a view of the primary raw reading, so it is published only while the
block's adopted text IS that reading: every span primary or agreement. An overlay edit changes
the text, so the structure is read again from the corrected text by the parser an engine's text
already goes through, and the engine's equation number and caption are kept where the text does
not carry them. A span left open leaves no structure at all: the block is its passage, markers
in place, beside its crop. The vault, the corpus, and the resolved score all ask here, so the
three agree on what was emitted.
"""

from typing import Literal

from llb.core.contracts.data_prep.book_blocks import Block, FormulaReading, RawReading, TableReading
from llb.core.contracts.data_prep.book_resolve import ResolvedBlock
from llb.prep.book.grid import formula_from_text, table_from_text
from llb.prep.book.resolve.technical import caption_contest

ResolutionState = Literal["verbatim", "edited", "open", "missing"]
VERBATIM: ResolutionState = "verbatim"
EDITED: ResolutionState = "edited"
OPEN: ResolutionState = "open"
MISSING: ResolutionState = "missing"
_AS_READ = frozenset({"primary", "agreement"})


def resolution_state(resolved: ResolvedBlock | None) -> ResolutionState:
    """`verbatim` as read, `edited` by an overlay, `open` while a span is unresolved."""
    if resolved is None:
        return MISSING
    provenances = {span.provenance for span in resolved.spans}
    if "unresolved" in provenances:
        return OPEN
    if provenances <= _AS_READ:
        return VERBATIM
    return EDITED


def emitted_structure(
    reading: RawReading | None, resolved: ResolvedBlock | None
) -> tuple[FormulaReading | None, TableReading | None]:
    """The formula and table this block publishes, or none where it abstains."""
    if reading is None:
        return None, None
    state = resolution_state(resolved)
    if state == VERBATIM:
        return reading.formula, reading.table
    if state != EDITED or resolved is None:
        return None, None
    return _edited_formula(reading, resolved.text), _edited_table(reading, resolved.text)


def _edited_formula(reading: RawReading, text: str) -> FormulaReading | None:
    if reading.formula is None:
        return None
    derived = formula_from_text(text, display=reading.formula.display)
    if derived is None:
        return None
    if derived.equation_number is None and reading.formula.equation_number:
        return derived.model_copy(
            update={
                "equation_number": reading.formula.equation_number,
                "number_side": reading.formula.number_side,
            }
        )
    return derived


def _edited_table(reading: RawReading, text: str) -> TableReading | None:
    if reading.table is None:
        return None
    return table_from_text(text, reading.table.caption)


def emitted_caption(
    block: Block, reading: RawReading | None, resolved: ResolvedBlock | None
) -> str:
    """The caption a table prints from its structure. A contested caption is not printed.

    Resolve already holds the whole table open when its captions disagree, so the crop beside
    the marker is where a reader sees it.
    """
    if reading is None or reading.table is None or not reading.table.caption:
        return ""
    if resolution_state(resolved) == MISSING or caption_contest(block, reading):
        return ""
    return reading.table.caption
