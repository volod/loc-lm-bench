"""A one-page book-ocr run for the review-queue tests."""

import json
from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookSourceRecord, BookStageSpec
from llb.core.contracts.data_prep.book_blocks import PageBlockLedger
from llb.core.contracts.data_prep.book_resolve import PageCorrectionOverlay, PageResolvedText
from llb.core.fsutil import atomic_write_text
from llb.prep.book.engines.protocol import EngineRead, Proposed
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.ledger.io import read_ledger, write_ledger
from llb.prep.book.preprocess.images import write_png
from llb.prep.book.resolve.apply import resolve_page
from llb.prep.book.resolve.io import decisions_path, read_decisions, write_overlay, write_resolved
from llb.prep.book.resolve.primary import PrimaryChoice

WIDTH = 200
HEIGHT = 400
_SHA = "ab" * 32


def write_run(root: Path, *reads: EngineRead, image: bool = True) -> Path:
    """A run directory holding one resolved page. `legacy` and `adapted` are the directory itself."""
    run = root if root.name in {"legacy", "adapted"} else root / "run"
    publish(run, assemble("p0001", 1, WIDTH, HEIGHT, reads), image=image)
    return run


def publish(run: Path, ledger: PageBlockLedger, *, image: bool) -> None:
    """Write config, the ledger, and an unresolved resolution of it."""
    _config(run)
    overlay, resolved = resolve_page(ledger, choice(), [])
    store(run, ledger, resolved, overlay, image=image)


def store(
    run: Path,
    ledger: PageBlockLedger,
    resolved: PageResolvedText,
    overlay: PageCorrectionOverlay,
    *,
    image: bool = False,
) -> None:
    """Replace the published page files. An image is written only when asked."""
    page = run / "pages" / ledger.page_id
    write_ledger(page / "blocks.json", ledger)
    write_resolved(page / "resolved.json", resolved)
    write_overlay(page / "overlay.json", overlay)
    if image:
        write_png(page / "render" / "page.png", np.zeros((HEIGHT, WIDTH), dtype=np.uint8))


def body(
    text: str,
    *,
    engine: str = "a",
    kind: str = "body",
    confidence: float | None = None,
    y0: float = 0.0,
    y1: float = 40.0,
) -> EngineRead:
    """One proposed block."""
    proposed = Proposed(kind, 0, 0.0, y0, 100.0, y1, text, confidence)
    return EngineRead(engine, "m", "original", "ocr", (proposed,))


def choice() -> PrimaryChoice:
    """The primary reading the fixture engines are named for."""
    return PrimaryChoice("ocr", "a", "original", None, "")


def read_page_ledger(run: Path) -> PageBlockLedger:
    """The published ledger of the fixture page."""
    return read_ledger(run / "pages" / "p0001" / "blocks.json")


def ledger_bytes(run: Path) -> bytes:
    """The raw ledger file, so a test can see that a decision did not rewrite it."""
    return (run / "pages" / "p0001" / "blocks.json").read_bytes()


def reresolve(run: Path) -> PageResolvedText:
    """Apply the run's reviewer file and publish the new resolution."""
    ledger = read_page_ledger(run)
    edits = list(read_decisions(decisions_path(run)).edits)
    _overlay, resolved = resolve_page(ledger, choice(), edits)
    store(run, ledger, resolved, _overlay)
    return resolved


def _config(run: Path) -> None:
    config = BookOcrRunConfig(
        schema_id="llb.book-ocr-run-config",
        schema_version="1.0.0",
        kind="book-ocr-run-config",
        run_id="review-test",
        created_at="2026-11-23T00:00:00Z",
        source=BookSourceRecord(
            original_path="book.pdf",
            sha256=_SHA,
            size_bytes=1,
            page_count=1,
            policy="symlink",
            stored_path="source/book.pdf",
        ),
        pages_spec="1",
        pages=[1],
        batch_pages=25,
        max_attempts=2,
        stages=[BookStageSpec(name="resolve")],
        reuse_from=None,
    )
    atomic_write_text(
        run / "config.json", json.dumps(config.model_dump(mode="json"), indent=2) + "\n"
    )
