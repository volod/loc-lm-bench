"""Ledger facts the report needs: open marks, orphans, engines, and confidence.

Markers are matched to the review queue when `review/queue.json` exists, and to
the same ledger marks the queue is built from when it does not. Orphaned
overlay entries are counted either way. A block whose recorded page disagrees
with the ledger that holds it is a failure.
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from llb.core.contracts.data_prep.book import BookOcrRunConfig
from llb.core.contracts.data_prep.book_resolve import (
    ORPHAN_REASONS,
    CorrectionEdit,
    PageCorrectionOverlay,
)
from llb.core.contracts.data_prep.book_review import ReviewQueueEntry
from llb.prep.book.cache import page_dir
from llb.prep.book.constants import BLOCKS_FILENAME
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.ranges import page_id
from llb.prep.book.resolve.constants import (
    OVERLAY_FILENAME,
    OVERLAY_KIND,
    OVERLAY_SCHEMA_ID,
    RESOLVED_FILENAME,
    SCHEMA_VERSION,
)
from llb.prep.book.resolve.io import read_overlay, read_resolved
from llb.prep.book.review.classify import page_marks
from llb.prep.book.review.files import reviewer_edits
from llb.prep.book.review.io import queue_path, read_queue
from llb.prep.book.review.marks import SpanMark
from llb.prep.book.validate.constants import BLOCK_WITHOUT_PAGE, CONFIDENCE_BUCKETS
from llb.prep.book.validate.types import EngineScores, Finding, OpenMark, RunFacts


def scan_run(run_dir: Path, config: BookOcrRunConfig) -> RunFacts:
    """Read every selected page once."""
    edits = reviewer_edits(run_dir)
    marks: list[OpenMark] = []
    findings: list[Finding] = []
    orphans: Counter[str] = Counter()
    scores: dict[str, EngineScores] = {}
    readings: Counter[tuple[str, str, str]] = Counter()
    kinds: dict[str, str] = {}
    for pdf_page in config.pages:
        _scan_page(
            run_dir, config, pdf_page, edits, marks, findings, orphans, scores, readings, kinds
        )
    queue = _queue_marks(run_dir, kinds)
    return RunFacts(
        marks=marks,
        queue_marks=queue,
        orphans=sum(orphans.values()),
        orphaned_by_reason={reason: orphans.get(reason, 0) for reason in ORPHAN_REASONS},
        scores=scores,
        readings=[(*key, count) for key, count in sorted(readings.items())],
        findings=findings,
    )


def _scan_page(
    run_dir: Path,
    config: BookOcrRunConfig,
    pdf_page: int,
    edits: Sequence[CorrectionEdit],
    marks: list[OpenMark],
    findings: list[Finding],
    orphans: Counter[str],
    scores: dict[str, EngineScores],
    readings: Counter[tuple[str, str, str]],
    kinds: dict[str, str],
) -> None:
    pid = page_id(pdf_page, config.source.page_count)
    base = page_dir(run_dir, pid)
    ledger_path = base / BLOCKS_FILENAME
    if ledger_path.is_file():
        _ledger(ledger_path, pid, findings, scores, readings, kinds)
    overlay = _overlay(base, pid, pdf_page)
    for item in overlay.orphaned:
        orphans[item.reason] += 1
    resolved_path = base / RESOLVED_FILENAME
    if not resolved_path.is_file() or not ledger_path.is_file():
        return
    ledger = read_ledger(ledger_path)
    resolved = read_resolved(resolved_path)
    blocks = {block.block_id: block for block in ledger.blocks}
    for mark in page_marks(pid, blocks, resolved, overlay, edits):
        if mark.verdict:
            continue
        marks.append(_from_span(mark, kinds))


def _ledger(
    path: Path,
    pid: str,
    findings: list[Finding],
    scores: dict[str, EngineScores],
    readings: Counter[tuple[str, str, str]],
    kinds: dict[str, str],
) -> None:
    ledger = read_ledger(path)
    rel = path.relative_to(path.parents[2]).as_posix()
    if ledger.page_id != pid:
        findings.append(
            Finding(BLOCK_WITHOUT_PAGE, rel, f"ledger page id is {ledger.page_id}, not {pid}")
        )
    for block in ledger.blocks:
        kinds[block.block_id] = block.kind
        if block.pdf_page != ledger.pdf_page or block.pdf_page < 1:
            findings.append(
                Finding(
                    BLOCK_WITHOUT_PAGE,
                    rel,
                    f"block {block.block_id} records PDF page {block.pdf_page} on {ledger.page_id}",
                )
            )
        for reading in block.readings:
            model = reading.model or "none"
            readings[(reading.engine, model, reading.variant)] += 1
            bucket = _bucket(reading.confidence)
            scores.setdefault(reading.engine, EngineScores()).add(reading.confidence, bucket)


def _from_span(mark: SpanMark, kinds: dict[str, str]) -> OpenMark:
    return OpenMark(
        kind=mark.kind,
        block_id=mark.block_id,
        start=mark.start,
        end=mark.end,
        pdf_page=mark.pdf_page,
        block_kind=kinds.get(mark.block_id, "absent"),
        classes=tuple(mark.classes),
    )


def _queue_marks(run_dir: Path, kinds: dict[str, str]) -> list[OpenMark] | None:
    path = queue_path(run_dir)
    if not path.is_file():
        return None
    return [_from_entry(entry, kinds) for entry in read_queue(path).entries if not entry.verdict]


def _from_entry(entry: ReviewQueueEntry, kinds: dict[str, str]) -> OpenMark:
    return OpenMark(
        kind=entry.kind,
        block_id=entry.block_id,
        start=entry.start,
        end=entry.end,
        pdf_page=entry.pdf_page,
        block_kind=kinds.get(entry.block_id, "absent"),
        classes=tuple(entry.classes),
    )


def _overlay(base: Path, pid: str, pdf_page: int) -> PageCorrectionOverlay:
    path = base / OVERLAY_FILENAME
    if path.is_file():
        return read_overlay(path)
    return PageCorrectionOverlay(
        schema_id=OVERLAY_SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        kind=OVERLAY_KIND,
        page_id=pid,
        pdf_page=pdf_page,
    )


def _bucket(score: float | None) -> str:
    if score is None:
        return "missing"
    for name, low, high in CONFIDENCE_BUCKETS:
        if low <= score < high:
            return name
    return CONFIDENCE_BUCKETS[-1][0]
