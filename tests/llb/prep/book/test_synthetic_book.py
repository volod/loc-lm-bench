"""One generated book, from the PDF through a validated vault and an ingested corpus.

A failure names the stage whose output the next stage could not use. The scan is
generated here; no page of a real book is an input.
"""

from collections.abc import Callable
from pathlib import Path

from llb.prep.book.classes.records import read_classes
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.ledger.manifest import read_manifest
from llb.prep.book.preprocess.records import read_signals
from llb.prep.book.resolve.io import read_resolved
from llb.prep.book.review.build import build_review_queue
from llb.prep.book.review.settled import terminal_failure
from llb.prep.book.runner import resume_run
from llb.prep.book.structure.build import build_structure
from tests.llb.prep.book.synthetic_book import (
    CHAPTER,
    COLUMNS,
    CONTENTS,
    DISPUTE,
    DISPUTE_TEXT,
    FAILED_PAGE,
    FORMULA_PAGE,
    GAP,
    KILL_PAGE,
    LEFT_COLUMN,
    PAGE_COUNT,
    RIGHT_COLUMN,
    TABLE_PAGE,
    UPSIDE_PAGE,
    begin,
    make_registry,
    write_scan,
)
from tests.llb.prep.book.synthetic_publish import _clean, _registry, _run, publish_steps

_State = dict[str, object]
_Step = Callable[[_State], None]


def test_synthetic_book_reaches_a_validated_corpus(tmp_path: Path) -> None:
    """The chain fails naming the stage whose output broke the next."""
    scan = write_scan(tmp_path / "library" / "scan.pdf")
    clean = begin(scan, tmp_path / "clean", make_registry(kill=False))
    killed = make_registry(kill=True)
    partial = begin(scan, tmp_path / "killed", killed)
    state: _State = {"scan": scan, "clean": clean, "run": partial, "registry": killed}
    previous = "scan"
    for name, step in _steps():
        try:
            step(state)
        except Exception as exc:
            raise AssertionError(f"{previous}: output broke {name}: {exc}") from exc
        previous = name


def _steps() -> tuple[tuple[str, _Step], ...]:
    return (
        ("resume", _resume),
        ("signals", _signals),
        ("classes", _classes),
        ("blocks", _blocks),
        ("resolve", _resolve),
        ("review-queue", _queue),
        ("structure", _structure),
        *publish_steps(),
    )


def _resume(state: _State) -> None:
    run = _run(state)
    page = run / "pages" / f"p{KILL_PAGE:04d}"
    assert not (page / "ocr").exists()
    assert (page / ".ocr.partial").is_dir()
    resume_run(run, registry=_registry(state))
    assert not (page / ".ocr.partial").exists()
    assert (page / "ocr").is_dir()
    clean = _pngs(_clean(state))
    assert _pngs(run) == clean
    manifest = read_manifest(run)
    assert manifest is not None and len(manifest.pages) == PAGE_COUNT
    assert terminal_failure(run, f"p{FAILED_PAGE:04d}")


def _signals(state: _State) -> None:
    run = _run(state)
    upside = read_signals(run / "pages" / f"p{UPSIDE_PAGE:04d}" / "signals.json")
    upright = read_signals(run / "pages" / f"p{FORMULA_PAGE:04d}" / "signals.json")
    assert upside.orientation_deg == 180
    assert upright.orientation_deg == 0


def _classes(state: _State) -> None:
    record = read_classes(_run(state) / "pages" / f"p{COLUMNS:04d}" / "classes.json")
    assert "multi_column" in record.classes


def _blocks(state: _State) -> None:
    run = _run(state)
    dispute = read_ledger(run / "pages" / f"p{DISPUTE:04d}" / "blocks.json")
    spans = [span for block in dispute.blocks for span in block.uncertainty]
    assert spans and any("digit" in span.classes for span in spans)
    columns = _joined(run, COLUMNS)
    assert columns.index(LEFT_COLUMN) < columns.index(RIGHT_COLUMN)
    formula = read_ledger(run / "pages" / f"p{FORMULA_PAGE:04d}" / "blocks.json")
    numbers = [block.readings[0].formula for block in formula.blocks]
    assert any(item is not None and item.equation_number == "2-1" for item in numbers)
    table = read_ledger(run / "pages" / f"p{TABLE_PAGE:04d}" / "blocks.json")
    assert any(block.readings[0].table is not None for block in table.blocks)
    assert not (run / "pages" / f"p{FAILED_PAGE:04d}" / "blocks.json").exists()


def _resolve(state: _State) -> None:
    page = read_resolved(_run(state) / "pages" / f"p{DISPUTE:04d}" / "resolved.json")
    assert any(span.provenance == "unresolved" for block in page.blocks for span in block.spans)
    assert DISPUTE_TEXT not in _joined(_run(state), DISPUTE)


def _queue(state: _State) -> None:
    run = _run(state)
    queue, _manifest = build_review_queue(run)
    open_entries = [entry for entry in queue.entries if not entry.verdict]
    failed = [entry for entry in open_entries if entry.reason == "page-failed"]
    assert any(entry.pdf_page == FAILED_PAGE for entry in failed)
    dispute = [
        entry
        for entry in open_entries
        if entry.pdf_page == DISPUTE and entry.reason == "unresolved"
    ]
    assert dispute and dispute[0].raw_text != dispute[0].candidate
    state["entry"] = queue.entries.index(dispute[0])


def _structure(state: _State) -> None:
    structure = build_structure(_run(state))
    assert len(structure.pages) == PAGE_COUNT
    printed = {page.pdf_page: (page.printed, page.status) for page in structure.pages}
    assert printed[FAILED_PAGE] == (None, "front-matter")
    assert printed[CONTENTS] == ("i", "observed")
    assert printed[CHAPTER] == ("10", "observed")
    assert printed[GAP] == ("12", "gap")
    assert printed[COLUMNS] == ("13", "observed")
    assert sum(page.status == "gap" for page in structure.pages) == 1
    titles = _titles(structure.hierarchy)
    assert {"1 Mission Design", "1.1 Orbit Window", "2 Propulsion", "Index"} <= titles
    assert structure.contents_pages == [CONTENTS]
    assert not structure.conflicts


def _joined(run: Path, pdf_page: int) -> str:
    page = read_resolved(run / "pages" / f"p{pdf_page:04d}" / "resolved.json")
    return "\n".join(block.text for block in page.blocks)


def _titles(node: object) -> set[str]:
    found = set()
    stack = [node]
    while stack:
        current = stack.pop()
        title = getattr(current, "title", "")
        number = getattr(current, "number", "")
        if title:
            found.add(f"{number} {title}".strip())
        stack.extend(getattr(current, "children", ()))
    return found


def _pngs(run: Path) -> dict[str, bytes]:
    return {
        path.parent.parent.name: path.read_bytes()
        for path in (run / "pages").glob("*/render/page.png")
    }
