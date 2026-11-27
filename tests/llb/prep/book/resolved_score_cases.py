"""Caption, formula, and unmapped fixtures for the resolved-lane score tests."""

from pathlib import Path

from llb.core.contracts.data_prep.book_blocks import Block
from llb.core.contracts.data_prep.book_resolve import ResolvedSpan
from llb.core.contracts.data_prep.book_structure import NumberedItem
from llb.prep.book.metrics.reference import RefPage
from llb.prep.book.ranges import page_id
from llb.prep.book.resolve.io import read_resolved, write_resolved
from tests.llb.prep.book.resolved_score_fixture import (
    PAGE_COUNT,
    _adopted,
    _body,
    _box,
    _mapping,
    _placed,
    _ref,
    _save,
    _withheld,
    _write_reference,
)


def hand_case(root: Path) -> tuple[Path, Path]:
    """Three reference pages, two engines, and one extra resolved page outside the reference.

    Page 1 adopts ``cat`` and withholds one symbol. Page 2 has a ledger and no resolved
    file. Page 3 adopts ``S`` against a reference ``5``. Page 9 is not in the reference.
    """
    run = root / "run"
    page1 = _body(1, "cat", deskew="dog")
    _save(run, [page1], _withheld(page1))
    _save(run, [_body(2, "cat")], None)
    _save(run, [_body(3, "S")], _adopted(_body(3, "S"), "S"))
    _save(run, [_body(9, "extra")], _adopted(_body(9, "extra"), "extra"))
    _mapping(run, (1, "1"), (2, "2"), (3, "3"), (9, "9"))
    reference = root / "reference"
    _write_reference(
        reference,
        [
            _ref(1, "prose", "1", [(_body(1, "cat").block_id, "body", "cat")]),
            _ref(2, "prose", "2", [(_body(2, "cat").block_id, "body", "cat")]),
            _ref(3, "glyph", "3", [(_body(3, "S").block_id, "body", "5")]),
        ],
    )
    return run, reference


def write_adopted(run: Path, pdf_page: int, text: str, variant: str) -> None:
    """Replace every adopted span on one page. The ledger is left as it was."""
    path = run / "pages" / page_id(pdf_page, PAGE_COUNT) / "resolved.json"
    page = read_resolved(path)
    blocks = [
        block.model_copy(
            update={
                "variant": variant,
                "primary_text": text,
                "text": text,
                "spans": [
                    ResolvedSpan(
                        provenance="primary", text=text, start=0, end=len(text), adopted=True
                    )
                ],
            }
        )
        for block in page.blocks
    ]
    write_resolved(path, page.model_copy(update={"blocks": blocks, "primary_variant": variant}))


def formula_case(root: Path) -> tuple[Path, Path]:
    """A formula's printed number comes from the inventory, and a pipe table stays a grid."""
    run = root / "run"
    formula = _placed(1, 0, "formula", "v_{E}", _box(10, 20, 120, 50))
    table_text = "| Maneuver | dv |\n| --- | --- |\n| Insertion | 5 |"
    table = _placed(1, 1, "table", table_text, _box(10, 60, 160, 120))
    adopted = [
        _adopted(formula, formula.readings[0].text),
        _adopted(table, table.readings[0].text),
    ]
    _save(run, [formula, table], adopted)
    _mapping(
        run,
        (1, "57"),
        items=[
            NumberedItem(
                kind="equation",
                number="(3-4)",
                pdf_page=1,
                printed="57",
                block_id=formula.block_id,
                evidence="display-number",
            )
        ],
    )
    reference = root / "reference"
    _write_reference(
        reference,
        [
            _ref(
                1,
                "formula",
                "57",
                [
                    (formula.block_id, "formula", "v_{E}"),
                    (table.block_id, "table", ""),
                ],
                [(10, 20, 120, 50), (10, 60, 160, 120)],
                equation_number="(3-4)",
                table_rows=[["Maneuver", "dv"], ["Insertion", "5"]],
            )
        ],
    )
    return run, reference


def caption_case(root: Path) -> tuple[Path, Path]:
    """The neighbour caption is not the printed one, and the mapped folio is wrong."""
    wrong = _placed(1, 2, "caption", "Fig. 9-9. Wrong.", _box(0, 100, 30, 120))
    right = _placed(1, 3, "caption", "Fig. 3-2. Right.", _box(20, 0, 50, 10))
    return _captioned(root, wrong, right)


def competing_case(root: Path) -> tuple[Path, Path]:
    """Two unnumbered captions one edge gap away. The vault attaches neither.

    The centre-distance guess still picks the right one, so a lane that guesses would pass.
    """
    wrong = _placed(1, 2, "caption", "Wrong label.", _box(0, 12, 40, 22))
    right = _placed(1, 3, "caption", "Right label.", _box(12, 0, 32, 10))
    return _captioned(root, wrong, right)


def _captioned(root: Path, wrong: Block, right: Block) -> tuple[Path, Path]:
    run = root / "run"
    figure = _placed(1, 1, "figure", "", _box(0, 0, 10, 10))
    body = _placed(1, 0, "body", "The folio is 57.", _box(10, 140, 120, 170))
    folio = _placed(1, 4, "page-number", "57", _box(80, 360, 120, 390))
    blocks = [body, figure, wrong, right, folio]
    _save(run, blocks, [_adopted(block, block.readings[0].text) for block in blocks])
    bare = _placed(2, 0, "body", "line", _box(10, 20, 80, 40))
    _save(run, [bare], [_adopted(bare, "line")])
    _mapping(run, (1, "99"), (2, "2"))
    reference = root / "reference"
    _write_reference(reference, [_caption_page(body, figure, wrong, right, folio), _bare_page()])
    return run, reference


def unmapped_case(root: Path) -> tuple[Path, Path]:
    """Resolved text with no document structure. The body names the reference folio."""
    run = root / "run"
    body = _body(1, "see 57")
    _save(run, [body], [_adopted(body, "see 57")])
    reference = root / "reference"
    _write_reference(reference, [_ref(1, "prose", "57", [(body.block_id, "body", "see 57")])])
    return run, reference


def _caption_page(body: Block, figure: Block, wrong: Block, right: Block, folio: Block) -> RefPage:
    return _ref(
        1,
        "figure",
        "57",
        [
            (body.block_id, "body", "The folio is 57."),
            (figure.block_id, "figure", "", right.block_id),
            (wrong.block_id, "caption", wrong.readings[0].text),
            (right.block_id, "caption", right.readings[0].text),
            (folio.block_id, "page-number", "57"),
        ],
        [
            (10, 140, 120, 170),
            (0, 0, 10, 10),
            _corners(wrong),
            _corners(right),
            (80, 360, 120, 390),
        ],
    )


def _corners(block: Block) -> tuple[float, float, float, float]:
    region = block.region
    return (region.x0, region.y0, region.x1, region.y1)


def _bare_page() -> RefPage:
    return _ref(
        2,
        "bare",
        "2",
        [
            ("p2-b0", "body", "line"),
            ("p2-b1", "formula", "x"),
            ("p2-b2", "table", ""),
            ("p2-b3", "figure", "", "p2-b4"),
            ("p2-b4", "caption", "Fig. 1-1."),
        ],
    )
