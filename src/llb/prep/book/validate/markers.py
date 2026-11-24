"""Open markers against the review queue, one entry each."""

from collections import Counter

from llb.prep.book.render.constants import TOC_FILENAME
from llb.prep.book.review.markers import parse_markers
from llb.prep.book.validate.constants import (
    MARKER_MISMATCH,
    QUEUE_DRIFT_WARNING,
    QUEUE_WARNING,
)
from llb.prep.book.validate.text import matter_int
from llb.prep.book.validate.types import BookContext, Finding, OpenMark, RunFacts

_Key = tuple[str, str, int, int, int]
_QUEUE_FILE = "review/queue.json"


def marker_check(
    ctx: BookContext, facts: RunFacts
) -> tuple[list[OpenMark], list[Finding], list[str]]:
    """Match vault markers to open queue entries, or to the ledger when the queue is absent."""
    vault = _vault_marks(ctx)
    if facts.queue_marks is not None:
        warnings = _drift(facts)
        missing_at = _QUEUE_FILE
        expected = facts.queue_marks
    else:
        warnings = [QUEUE_WARNING]
        missing_at = ""
        expected = facts.marks
    findings = _diff(vault, expected, ctx, missing_at)
    return expected, findings, warnings


def _drift(facts: RunFacts) -> list[str]:
    queue = Counter(mark.key for mark in facts.queue_marks or [])
    ledger = Counter(mark.key for mark in facts.marks)
    if queue == ledger:
        return []
    return [QUEUE_DRIFT_WARNING]


def _vault_marks(ctx: BookContext) -> list[tuple[_Key, str]]:
    found: list[tuple[_Key, str]] = []
    docs = [*ctx.content]
    if ctx.toc is not None:
        docs.append(ctx.toc)
    for doc in docs:
        for marker in parse_markers(doc.body):
            page = _page(marker.pdf_link, ctx.fragment_name)
            key = (marker.kind, marker.block_id, marker.start, marker.end, page)
            found.append((key, doc.rel))
    return found


def _diff(
    vault: list[tuple[_Key, str]],
    expected: list[OpenMark],
    ctx: BookContext,
    missing_at: str,
) -> list[Finding]:
    have = Counter(key for key, _file in vault)
    want = Counter(mark.key for mark in expected)
    files = {key: file for key, file in vault}
    findings: list[Finding] = []
    for key, count in sorted(have.items()):
        if count > want.get(key, 0):
            detail = f"open marker {_label(key)} has no matching review-queue entry"
            findings.append(Finding(MARKER_MISMATCH, files[key], detail))
    for key, count in sorted(want.items()):
        if count > have.get(key, 0):
            file = missing_at or _cover(ctx, key[4])
            findings.append(
                Finding(
                    MARKER_MISMATCH, file, f"open review-queue entry {_label(key)} has no marker"
                )
            )
    return findings


def _cover(ctx: BookContext, page: int) -> str:
    for doc in ctx.content:
        start = matter_int(doc.matter, "pdf_start")
        end = matter_int(doc.matter, "pdf_end")
        if start is not None and end is not None and start <= page <= end:
            return doc.rel
    return TOC_FILENAME


def _page(url: str, fragment_name: str) -> int:
    fragment = url.partition("#")[2]
    key, sep, value = fragment.partition("=")
    if key == fragment_name and sep and value.isdigit():
        return int(value)
    return -1


def _label(key: _Key) -> str:
    kind, block_id, start, end, page = key
    return f"{kind} {block_id} {start}-{end} on PDF {page}"
