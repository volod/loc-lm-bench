"""The resolved pipeline lane: the text, tables, formulas, links, and folio the renderers use.

Raw engine lanes stay beside it. This lane reads ``resolved.json`` for adopted text and
unresolved spans, the block ledger for kind and region, and the document structure for the
printed-page mapping and equation numbers. A formula or a table is scored as the vault emits it,
and caption links are the figure-asset association the vault prints.
A page with no resolved file is a missing submission. A structural metric the page did not
emit is absent, not a zero.
"""

from pathlib import Path

from pydantic import ValidationError

from llb.core.contracts.data_prep.book_blocks import (
    UNCERTAINTY_CLASSES,
    Block,
    FormulaReading,
    PageBlockLedger,
    RawReading,
    TableReading,
)
from llb.core.contracts.data_prep.book_resolve import PageResolvedText, ResolvedBlock
from llb.core.contracts.data_prep.book_structure import DocumentStructure
from llb.prep.book.grid import cells_of
from llb.prep.book.constants import BLOCKS_FILENAME, PAGES_DIRNAME
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.metrics.confusion import equation_numbers
from llb.prep.book.metrics.constants import (
    FIELD_CAPTION,
    FIELD_FIGURE,
    FIELD_FORMULA,
    FIELD_PRINTED,
    FIELD_TABLE,
    LANE_RESOLVED,
)
from llb.prep.book.metrics.geometry import Box
from llb.prep.book.metrics.resolved_captions import emitted_captions
from llb.prep.book.metrics.submission import HypBlock, PageSubmission
from llb.prep.book.metrics.tables import Cell, parse_table_text
from llb.prep.book.render.names import equation_numbers as inventory_numbers
from llb.prep.book.render.emitted import (
    OPEN,
    ResolutionState,
    emitted_structure,
    resolution_state,
)
from llb.prep.book.render.mathform import (
    IMAGE,
    formula_math,
    render_body,
    table_markdown,
    table_mode,
)
from llb.prep.book.resolve.constants import RESOLVED_FILENAME
from llb.prep.book.resolve.io import read_resolved
from llb.prep.book.resolve.primary import resolved_reading
from llb.prep.book.structure.io import read_structure, structure_path

_STRUCTURED = frozenset({"formula", "table"})
_KIND_FIELDS = (
    ("formula", FIELD_FORMULA),
    ("table", FIELD_TABLE),
    ("figure", FIELD_FIGURE),
)


def load_resolved(run_dir: Path) -> list[PageSubmission]:
    """One submission per resolved page. An empty list when the run has not resolved."""
    paths = sorted((run_dir / PAGES_DIRNAME).glob(f"*/{RESOLVED_FILENAME}"))
    if not paths:
        return []
    structure = _structure(run_dir)
    read = [_page(path) for path in paths]
    captions = emitted_captions(read, structure)
    pages = [_submission(ledger, resolved, structure, captions) for ledger, resolved in read]
    _unique(pages)
    return pages


def _structure(run_dir: Path) -> DocumentStructure | None:
    path = structure_path(run_dir)
    if not path.is_file():
        return None
    return read_structure(path)


def _page(path: Path) -> tuple[PageBlockLedger, PageResolvedText]:
    resolved = read_resolved(path)
    ledger_path = path.parent / BLOCKS_FILENAME
    if not ledger_path.is_file():
        raise ValueError(f"{resolved.page_id} has resolved text and no block ledger")
    try:
        ledger = read_ledger(ledger_path)
    except (OSError, ValidationError, ValueError) as exc:
        raise ValueError(f"{ledger_path} is not a readable block ledger") from exc
    if ledger.pdf_page != resolved.pdf_page:
        raise ValueError(
            f"{resolved.page_id} ledger is pdf page {ledger.pdf_page}, "
            f"resolved text is pdf page {resolved.pdf_page}"
        )
    return ledger, resolved


def _submission(
    ledger: PageBlockLedger,
    resolved: PageResolvedText,
    structure: DocumentStructure | None,
    captions: dict[str, str],
) -> PageSubmission:
    blocks = _blocks(ledger, resolved, structure, captions)
    folio_known, folio = _folio(resolved.pdf_page, structure)
    return PageSubmission(
        LANE_RESOLVED,
        resolved.pdf_page,
        tuple(blocks),
        False,
        unsupported=_unsupported(blocks, folio_known),
        uses_emitted_folio=folio_known,
        emitted_folio=folio,
        abstentions=tuple(_abstentions(resolved).items()),
    )


def _blocks(
    ledger: PageBlockLedger,
    resolved: PageResolvedText,
    structure: DocumentStructure | None,
    links: dict[str, str],
) -> list[HypBlock]:
    by_id = {block.block_id: block for block in resolved.blocks}
    ordered = sorted(ledger.blocks, key=lambda block: (block.reading_order, block.ordinal))
    numbers = inventory_numbers(structure) if structure is not None else {}
    built = [_hyp_block(block, by_id.get(block.block_id), links, numbers) for block in ordered]
    seen = {block.block_id for block in ordered}
    for extra in resolved.blocks:
        if extra.block_id not in seen:
            built.append(_orphan(extra))
    return built


def _hyp_block(
    block: Block,
    resolved: ResolvedBlock | None,
    links: dict[str, str],
    numbers: dict[str, str],
) -> HypBlock:
    text = resolved.text if resolved is not None else ""
    reading = resolved_reading(block, resolved)
    formula, table = emitted_structure(reading, resolved)
    number = numbers.get(block.block_id, "") if block.kind == "formula" else ""
    found = equation_numbers(text) if block.kind == "formula" else []
    if formula is not None and formula.equation_number:
        number = formula.equation_number
    equation = number or (found[0] if found else "")
    region = block.region
    state = resolution_state(resolved)
    return HypBlock(
        block_id=block.block_id,
        kind=block.kind,
        region=Box(region.x0, region.y0, region.x1, region.y1),
        reading_order=block.reading_order,
        text=text,
        latex=_formula(block.kind, text, number, formula, state),
        table=_table(block.kind, text, table, reading),
        caption_id=links.get(block.block_id) if block.kind == "figure" else None,
        equation_number=equation or None,
        held_open=block.kind in _STRUCTURED and state == OPEN,
    )


def _orphan(resolved: ResolvedBlock) -> HypBlock:
    return HypBlock(
        block_id=resolved.block_id,
        kind="unknown",
        region=None,
        reading_order=10**6,
        text=resolved.text,
        latex=None,
        table=None,
        caption_id=None,
        equation_number=None,
    )


def _formula(
    kind: str, text: str, number: str, formula: FormulaReading | None, state: ResolutionState
) -> str | None:
    """The math the vault emits. A formula held open emits its passage, not math."""
    if kind != "formula" or state == OPEN:
        return None
    source = text
    if formula is not None:
        source = render_body(formula.latex) or formula.latex
    rendered = formula_math(source, number)
    if rendered.startswith("$$") and rendered.endswith("$$"):
        return rendered[2:-2].strip()
    return rendered.strip()


def _table(
    kind: str, text: str, table: TableReading | None, reading: RawReading | None
) -> tuple[tuple[Cell, ...], ...] | None:
    """The grid the vault emits: the published structure, the page image, or the passage."""
    if kind != "table":
        return None
    if table is not None:
        if table_mode(table) == IMAGE:
            return None
        return cells_of(table)
    if reading is not None and reading.table is not None and table_mode(reading.table) == IMAGE:
        return None
    if not text.strip():
        return None
    return parse_table_text(table_markdown(text))


def _folio(pdf_page: int, structure: DocumentStructure | None) -> tuple[bool, str | None]:
    if structure is None:
        return False, None
    for page in structure.pages:
        if page.pdf_page == pdf_page:
            return True, page.printed
    return False, None


def _unsupported(blocks: list[HypBlock], folio_known: bool) -> frozenset[str]:
    kinds = {block.kind for block in blocks}
    missing = [field for kind, field in _KIND_FIELDS if kind not in kinds]
    if "figure" not in kinds:
        missing.append(FIELD_CAPTION)
    if not folio_known:
        missing.append(FIELD_PRINTED)
    return frozenset(missing)


def _abstentions(resolved: PageResolvedText) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(UNCERTAINTY_CLASSES, 0)
    for block in resolved.blocks:
        for span in block.spans:
            if span.provenance != "unresolved":
                continue
            for name in span.classes:
                counts[name] += 1
    return counts


def _unique(pages: list[PageSubmission]) -> None:
    seen: set[int] = set()
    for page in pages:
        if page.pdf_page in seen:
            raise ValueError(f"resolved lane has two submissions for pdf page {page.pdf_page}")
        seen.add(page.pdf_page)
