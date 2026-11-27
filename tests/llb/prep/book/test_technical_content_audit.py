"""Technical values traced from two engines' readings through resolve to both renderings.

One planted run covers each case of the audit. A value no reading settles stays out of the
adopted text and is a marker in the vault, a token in the corpus, and a queue entry; the
validator finds the same set. The raw ledger is never rewritten. The score reads the formula,
grid, and caption association the vault actually emits.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from llb.core.contracts.data_prep.book_resolve import PageCorrectionOverlay, PageResolvedText
from llb.prep.book.ledger.io import ledger_path, read_ledger
from llb.prep.book.metrics.resolved import load_resolved
from llb.prep.book.project.build import project_corpus
from llb.prep.book.render.build import RenderOptions, build_vault
from llb.prep.book.resolve.constants import AGREEMENT_EVIDENCE
from llb.prep.book.resolve.io import read_overlay, read_resolved
from llb.prep.book.review.build import build_review_queue
from llb.prep.book.structure.build import build_structure
from llb.prep.book.structure.io import read_structure, structure_path
from llb.prep.book.validate.check import validate_book
from tests.llb.prep.book.technical_fixture import (
    AGREED_FORMULA,
    BODY,
    CITING,
    planted_pdf,
    recover,
)

_PAGES = ("p0001", "p0002", "p0003", "p0004")


@dataclass(frozen=True)
class Book:
    run: Path
    vault_text: str
    corpus_text: str
    ledgers: dict[str, bytes]
    queue: list[tuple[str, int, int, tuple[str, ...]]]
    vault: Path


def publish(run: Path, root: Path) -> Book:
    """Structure, queue, vault, and corpus over a finished run."""
    build_structure(run)
    queue, _manifest = build_review_queue(run)
    vault = build_vault(run, RenderOptions(output=root / "vault"))
    corpus = root / "corpus"
    project_corpus(run, corpus, "redistributable", language="en")
    rows = [(row.block_id, row.start, row.end, tuple(row.classes)) for row in queue.entries]
    return Book(run, _text(vault, ".md"), _text(corpus, ".md"), ledger_bytes(run), rows, vault)


def ledger_bytes(run: Path) -> dict[str, bytes]:
    return {pid: ledger_path(run, pid).read_bytes() for pid in _PAGES}


def resolved(run: Path, pid: str) -> PageResolvedText:
    return read_resolved(run / "pages" / pid / "resolved.json")


def overlay(run: Path, pid: str) -> PageCorrectionOverlay:
    return read_overlay(run / "pages" / pid / "overlay.json")


def block_by_text(run: Path, pid: str, prefix: str):
    page = resolved(run, pid)
    return next(block for block in page.blocks if block.primary_text.startswith(prefix))


def _text(root: Path, suffix: str) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(root.rglob(f"*{suffix}")))


@pytest.fixture(scope="module")
def book(tmp_path_factory: pytest.TempPathFactory) -> Book:
    root = tmp_path_factory.mktemp("technical")
    run = recover(planted_pdf(root / "library" / "planted.pdf"), root / "runs")
    before = ledger_bytes(run)
    built = publish(run, root)
    assert built.ledgers == before
    return built


def test_raw_readings_stay_immutable_and_every_edit_names_its_evidence(book: Book) -> None:
    body = next(
        block
        for block in read_ledger(ledger_path(book.run, "p0001")).blocks
        if block.readings[0].text == BODY
    )
    assert {reading.engine: reading.text for reading in body.readings} == {
        "alpha": BODY,
        "beta": "The burn lasts 0.05 s at x = 3.",
    }
    for pid in _PAGES:
        _assert_edits_name_evidence(overlay(book.run, pid))
        _assert_spans_are_primary_slices(resolved(book.run, pid))


def _assert_edits_name_evidence(page: PageCorrectionOverlay) -> None:
    for edit in page.applied:
        assert edit.evidence and edit.source
        assert edit.author != "agreement" or edit.evidence == AGREEMENT_EVIDENCE


def _assert_spans_are_primary_slices(page: PageResolvedText) -> None:
    for block in page.blocks:
        for span in block.spans:
            assert span.adopted == (span.provenance != "unresolved")
            if span.provenance in ("primary", "agreement", "unresolved"):
                assert block.primary_text[span.start : span.end] == span.text


def test_an_ambiguous_index_and_sign_stay_open_in_both_renderings(book: Book) -> None:
    body = block_by_text(book.run, "p0001", "The burn")
    held = [(span.text, span.classes) for span in body.spans if span.provenance == "unresolved"]
    # The primary shortened 0.05 and kept a minus the other engine did not read.
    assert held == [(".5", ["digit"]), ("-", ["sign"])]
    assert body.text == "The burn lasts 0 s at x = 3."
    for unsupported in ("0.5 s", "0.05 s", "x = -3"):
        assert unsupported not in book.vault_text and unsupported not in book.corpus_text
    formula = block_by_text(book.run, "p0001", "v_1")
    rows = [row for row in book.queue if row[0] == formula.block_id]
    assert [(row[1], row[2]) for row in rows] == [(2, 3), (4, 5), (8, 9)]
    assert [set(row[3]) for row in rows] == [{"digit", "symbol"}, {"sign"}, {"symbol"}]
    for text in (book.vault_text, book.corpus_text):
        assert "v_1" not in text and "v_i" not in text and "v_l" not in text
    assert f"review/{formula.block_id}-formula.png" in book.vault_text
    assert "$$\nv_" not in book.vault_text
    agreed = r"E = mc^2 \tag{2-5}"
    assert agreed in book.vault_text and agreed in book.corpus_text
    assert AGREED_FORMULA not in book.vault_text


def test_a_disputed_cell_is_a_marker_in_the_grid_and_a_merge_is_not_a_dispute(book: Book) -> None:
    disputed = block_by_text(book.run, "p0002", "Item")
    marker = f'"{disputed.block_id} 37-39"'
    assert "| Tank | 1200 |" in book.vault_text
    assert any(
        line.startswith("| Valve | [!formula-table]") and marker in line
        for line in book.vault_text.splitlines()
    )
    assert (
        f"| Valve | {{{{llb-marker:unresolved:{disputed.block_id}:37-39}}}} |" in book.corpus_text
    )
    for value in ("| 35 |", "| 36 |", "Valve | 35", "Valve | 36"):
        assert value not in book.vault_text and value not in book.corpus_text
    merged = next(
        block
        for block in read_ledger(ledger_path(book.run, "p0002")).blocks
        if block.readings[0].text.startswith("Stage 1")
    )
    assert merged.cell_disputes == []
    assert '<th colspan="2">Stage 1</th>' in book.vault_text
    omitted = block_by_text(book.run, "p0002", "Valve")
    assert [(span.start, span.end) for span in omitted.spans] == [(0, len(omitted.primary_text))]
    assert omitted.text == ""
    assert "| V1 |" not in book.vault_text and "Count" not in book.vault_text
    assert f"review/{omitted.block_id}-table.png" in book.vault_text
    captioned = block_by_text(book.run, "p0002", "a | b")
    assert next(row for row in book.queue if row[0] == captioned.block_id)[3] == ("digit",)
    for caption in ("Table 2-1", "Table 2-7"):
        assert caption not in book.vault_text and caption not in book.corpus_text


def test_figure_captions_and_numbers_come_only_from_what_the_page_prints(book: Book) -> None:
    record = json.loads((book.vault / "metadata" / "figure_assets.json").read_text("utf-8"))
    assert len(record["figures"]) == 3
    assert {(row["number"], row["caption_block_id"]) for row in record["figures"]} == {(None, None)}
    assert all("12" not in row["asset"] for row in record["figures"])
    inventory = read_structure(structure_path(book.run)).inventory
    assert [item for item in inventory if item.kind == "figure"] == []
    assert CITING in book.vault_text and "^fig-3-1" not in book.vault_text
    lines = book.vault_text.splitlines()
    assert "Burn sequence." in lines and "Thrust profile." in lines
    caption = block_by_text(book.run, "p0003", "Figure 12")
    assert f'"{caption.block_id} 10-11"' in book.vault_text
    assert "Figure 12-3" not in book.vault_text and "Figure 12-8" not in book.vault_text


def test_a_fallback_reading_is_the_route_every_output_follows(book: Book) -> None:
    fallback = block_by_text(book.run, "p0004", "F = m a")
    assert fallback.fallback is True and fallback.engine == "beta"
    shown = r"F = m a \tag{4-1}"
    assert shown in book.vault_text and shown in book.corpus_text
    scored = next(page for page in load_resolved(book.run) if page.pdf_page == 4)
    formula = next(block for block in scored.blocks if block.block_id == fallback.block_id)
    assert formula.latex == shown


def test_the_score_reads_what_the_vault_emits(book: Book) -> None:
    pages = {page.pdf_page: page for page in load_resolved(book.run)}
    first = {block.block_id: block for block in pages[1].blocks}
    held = first[block_by_text(book.run, "p0001", "v_1").block_id]
    assert held.latex is None and held.held_open is True
    agreed = first[block_by_text(book.run, "p0001", "E = mc").block_id]
    assert agreed.latex == r"E = mc^2 \tag{2-5}" and agreed.held_open is False
    second = {block.block_id: block for block in pages[2].blocks}
    merged = second[block_by_text(book.run, "p0002", "Stage 1").block_id]
    assert merged.table is not None and merged.table[0][0].colspan == 2
    whole = second[block_by_text(book.run, "p0002", "Valve").block_id]
    assert whole.held_open is True and whole.table is None
    figures = [block for block in pages[3].blocks if block.kind == "figure"]
    assert len(figures) == 3 and {block.caption_id for block in figures} == {None}


def test_every_marker_has_its_queue_entry(book: Book) -> None:
    report = validate_book(book.vault, book.run)
    assert report.findings == ()
    assert len(book.queue) == 9
