"""The seams a technical value crosses, each pinned on its own small input."""

from llb.core.contracts.data_prep.book_blocks import RawReading
from llb.prep.book.engines.protocol import EngineRead, Proposed
from llb.prep.book.grid import formula_from_text, table_from_grid
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.ledger.cells import cell_disputes
from llb.prep.book.render.emitted import EDITED, OPEN, VERBATIM, emitted_structure, resolution_state
from llb.prep.book.render.grammar import load_grammar
from llb.prep.book.resolve.anchor import block_anchor
from llb.prep.book.resolve.apply import resolve_page
from llb.prep.book.resolve.numbers import number_text
from llb.prep.book.resolve.primary import PrimaryChoice
from llb.prep.book.review.alternates import candidate_text
from llb.prep.book.structure.inventory import inventory_of
from llb.prep.book.structure.lines import Line
from tests.llb.prep.book.test_resolve import _edit

_CHOICE = PrimaryChoice("ocr", "a", "original", None, "")


def _ledger(kind: str, *texts: str, formula: bool = False):
    reads = []
    for engine, text in zip("abc", texts, strict=False):
        structure = formula_from_text(text, display=True) if formula else None
        block = Proposed(kind, 0, 0.0, 0.0, 100.0, 40.0, text, None, structure, None)
        reads.append(EngineRead(engine, "m", "original", "ocr", (block,)))
    return assemble("p0001", 1, 200, 400, reads)


def test_a_value_the_primary_shortened_is_held_with_the_longer_reading_as_candidate() -> None:
    ledger = _ledger("body", "lasts 0.5 s", "lasts 0.05 s")
    _overlay, page = resolve_page(ledger, _CHOICE, [])
    held = [span for span in page.blocks[0].spans if span.provenance == "unresolved"]
    assert [(span.text, span.classes) for span in held] == [(".5", ["digit"])]
    assert page.blocks[0].text == "lasts 0 s"
    block = ledger.blocks[0]
    assert candidate_text(block, block.readings[0], held[0].start, held[0].end) == ".05"


def test_a_dropped_minus_is_held_and_a_letter_in_prose_is_not() -> None:
    _overlay, signed = resolve_page(_ledger("body", "x = 3", "x = -3"), _CHOICE, [])
    assert [span.classes for span in signed.blocks[0].spans if not span.adopted] == [["sign"]]
    _overlay, prose = resolve_page(_ledger("body", "cat", "cot"), _CHOICE, [])
    assert all(span.adopted for span in prose.blocks[0].spans)


def test_a_formula_letter_another_formula_reading_writes_differently_stays_open() -> None:
    ledger = _ledger("formula", r"v_i = 0", r"v_j = 0", formula=True)
    _overlay, page = resolve_page(ledger, _CHOICE, [])
    held = [(span.text, span.classes) for span in page.blocks[0].spans if not span.adopted]
    assert held == [("i", ["symbol"])]
    assert resolution_state(page.blocks[0]) == OPEN
    assert emitted_structure(ledger.blocks[0].readings[0], page.blocks[0]) == (None, None)


def test_grids_are_compared_on_their_expanded_positions() -> None:
    merged = _table("a", [[{"text": "Mass 2", "col_span": 2}], ["1", "2"]])
    split = _table("b", [["Mass 2", ""], ["1", "2"]])
    assert cell_disputes([merged, split]) == []
    narrow = _table("a", [["Valve", "Mass"], ["V1", ""]])
    wide = _table("b", [["Valve", "Mass", "Count"], ["V1", "12", "4"]])
    found = {(item.row, item.column): item.classes for item in cell_disputes([narrow, wide])}
    assert found == {(1, 1): ["digit"], (1, 2): ["digit"]}


def test_a_number_an_unresolved_span_touches_is_never_read_across_the_gap() -> None:
    ledger = _ledger("caption", "Figure 12-3. Orbit", "Figure 12-8. Orbit")
    _overlay, page = resolve_page(ledger, _CHOICE, [])
    assert page.blocks[0].text == "Figure 12-. Orbit"
    assert number_text(page.blocks[0]) == "Figure ?. Orbit"
    _overlay, folio = resolve_page(_ledger("page-number", "125", "1S5"), _CHOICE, [])
    assert folio.blocks[0].text == "15" and number_text(folio.blocks[0]) == "?"


def test_a_sentence_citing_a_figure_is_not_its_caption() -> None:
    lines = [
        _line("Figure 3-1 shows the transfer."),
        _line("Figure 3-2. Orbit"),
        _line("Fig. 3-3 Burn"),
    ]
    numbers = [item.number for item in inventory_of(lines, {1: "1"}) if item.kind == "figure"]
    assert numbers == ["3-2", "3-3"]


def test_a_citation_that_runs_into_a_marker_is_not_a_citation() -> None:
    phrase = next(kind for kind in load_grammar(None).kinds if kind.name == "figure").phrase
    assert phrase.search("see Figure 12-3.") is not None
    vault = 'see Figure 12-[!low-confidence](book.pdf#page=3 "p0003-b0001 10-11").'
    corpus = "see Figure 12-{{llb-marker:unresolved:p0003-b0001:10-11}}."
    assert phrase.search(vault) is None and phrase.search(corpus) is None


def test_a_formula_is_published_as_read_as_corrected_or_not_at_all() -> None:
    ledger = _ledger("formula", "E = mc^2 (2-5)", formula=True)
    block = ledger.blocks[0]
    _overlay, page = resolve_page(ledger, _CHOICE, [])
    assert resolution_state(page.blocks[0]) == VERBATIM
    assert emitted_structure(block.readings[0], page.blocks[0])[0] == block.readings[0].formula
    _overlay, edited = resolve_page(ledger, _CHOICE, [_edit(block, "reviewer", "3", 7, 8)])
    assert resolution_state(edited.blocks[0]) == EDITED
    formula, _table_ = emitted_structure(block.readings[0], edited.blocks[0])
    assert formula is not None and formula.latex == "E = mc^3"
    assert formula.equation_number == "2-5"
    assert block_anchor(block) == edited.blocks[0].anchor


def _table(engine: str, rows) -> RawReading:
    grid = table_from_grid(rows)
    text = "\n".join(
        " | ".join(cell["text"] if isinstance(cell, dict) else cell for cell in row) for row in rows
    )
    return RawReading(engine=engine, variant="original", text=text, table=grid)


def _line(text: str) -> Line:
    return Line("p0001-b0000", 1, text, "body", 10.0, 100.0, 300.0, 120.0, 600, 800)
