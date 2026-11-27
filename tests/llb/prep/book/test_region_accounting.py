"""A finished run keeps a blank, an unread column, and a failed page apart."""

import json
from pathlib import Path

from llb.core.contracts.data_prep.book_review import ReviewQueueEntry
from llb.prep.book.ledger.io import ledger_path, read_ledger
from llb.prep.book.review.io import queue_path, read_manifest, read_queue, write_queue
from tests.llb.prep.book.region_fixture import (
    AFTER,
    BEFORE,
    BLANK,
    COLUMNS,
    EMPTY,
    FAILED,
    RIGHT,
    TEXT,
    command,
    planted_pdf,
    recover,
)


def test_four_outcomes_survive_the_run_queue_and_render(tmp_path: Path) -> None:
    pdf = tmp_path / "library" / "planted.pdf"
    planted_pdf(pdf)
    run = recover(pdf, tmp_path / "runs")
    left_edge = _left_edge(run)
    missed = _queue_rows(run, left_edge)
    vault = tmp_path / "vault"
    _render(run, vault)
    corpus = tmp_path / "corpus"
    _project(run, corpus, left_edge)
    _reject_a_broken_queue(run, vault, missed)


def _left_edge(run: Path) -> float:
    manifest = read_manifest(run / "metadata" / "quality_manifest.json")
    assert [page.coverage for page in manifest.pages] == [
        "read",
        "omission",
        "omission",
        "blank",
        "failed",
        "read",
    ]
    ledger = read_ledger(ledger_path(run, COLUMNS))
    assert ledger.blocks
    confidences = [reading.confidence or 0 for block in ledger.blocks for reading in block.readings]
    assert min(confidences) >= 0.99
    return max(block.region.x1 for block in ledger.blocks)


def _queue_rows(run: Path, left_edge: float) -> ReviewQueueEntry:
    queue = read_queue(queue_path(run))
    omissions = [entry for entry in queue.entries if entry.reason == "omission"]
    assert {entry.page_id for entry in omissions} == {EMPTY, COLUMNS}
    missed = next(entry for entry in omissions if entry.page_id == COLUMNS)
    assert missed.box
    assert missed.box[0] > left_edge
    assert missed.crop
    assert not any(entry.page_id == BLANK for entry in queue.entries)
    failed = next(entry for entry in queue.entries if entry.reason == "page-failed")
    assert failed.page_id == FAILED
    assert failed.kind == "page-failed"
    return missed


def _render(run: Path, vault: Path) -> None:
    rendered = command("book-ocr", "render", str(run), "--output", str(vault), "--slug", "planted")
    assert rendered.exit_code == 0, rendered.output
    prose = _markdown(vault)
    _keeps(prose)
    assert RIGHT not in prose
    assert "[!blank]" in prose
    assert "[!unreadable]" in prose
    assert "[!page-failed]" in prose
    report = json.loads((vault / "metadata" / "validation.json").read_text(encoding="utf-8"))
    assert len(report["pages"]) == 6
    assert {row["status"] for row in report["pages"]} == {"completed", "failed"}


def _project(run: Path, corpus: Path, left_edge: float) -> None:
    projected = command(
        "book-ocr",
        "project",
        str(run),
        "--out",
        str(corpus),
        "--licence",
        "redistributable",
        "--language",
        "en",
        "--slug",
        "planted",
    )
    assert projected.exit_code == 0, projected.output
    text = _markdown(corpus)
    _keeps(text)
    assert RIGHT not in text
    assert "{{llb-marker:omission:" in text
    assert "{{llb-marker:page-failed:" in text
    assert "{{llb-marker:blank:" in text
    _citation(corpus, left_edge)


def _reject_a_broken_queue(run: Path, vault: Path, missed: ReviewQueueEntry) -> None:
    saved = queue_path(run).read_bytes()
    queue_path(run).unlink()
    missing = command("book-ocr", "validate", str(vault), "--run", str(run))
    assert missing.exit_code == 1
    assert "missing-review-queue" in missing.output
    queue_path(run).write_bytes(saved)
    stale_queue = read_queue(queue_path(run))
    kept = [entry for entry in stale_queue.entries if entry.entry_id != missed.entry_id]
    write_queue(queue_path(run), stale_queue.model_copy(update={"entries": kept}))
    stale = command("book-ocr", "validate", str(vault), "--run", str(run))
    assert stale.exit_code == 1
    assert "stale-review-queue" in stale.output


def _keeps(text: str) -> None:
    for sentence in (TEXT[BEFORE], TEXT[COLUMNS], TEXT[AFTER]):
        assert sentence in text


def _citation(corpus: Path, left_edge: float) -> None:
    pages: set[int] = set()
    for path in corpus.glob("*.citations.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages.update(_omission_pages(payload, left_edge))
    assert pages == {2, 3}


def _omission_pages(payload: dict, left_edge: float) -> set[int]:
    found: set[int] = set()
    for page in payload["pages"]:
        found.update(_page_omissions(page, left_edge))
    return found


def _page_omissions(page: dict, left_edge: float) -> set[int]:
    found: set[int] = set()
    for block in page["blocks"]:
        reasons = {marker["reason"] for marker in block["markers"]}
        if "omission" not in reasons:
            continue
        found.add(page["page"])
        _right_of_left(page, block, left_edge)
    return found


def _right_of_left(page: dict, block: dict, left_edge: float) -> None:
    if page["page"] == 3:
        assert block["region"][0] > left_edge


def _markdown(root: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(root.rglob("*.md")))
