"""Scanned-book uncertainty queue. Decisions become correction-overlay edits."""

from pathlib import Path

from llb.core.contracts.data_prep.book_review import ReviewQueueEntry
from llb.prep.book.review.constants import (
    ACCEPT_CANDIDATE,
    ACCEPT_RAW,
    CLEAR_DECISION,
    ENTER_TEXT,
    MARK_ILLEGIBLE,
)
from llb.prep.book.review.decide import apply_review_decision
from llb.prep.book.review.io import read_queue
from llb.review.core import ReviewAction, ReviewAdapter, ReviewRecord
from llb.review.presentation import fields_section

_ACTIONS = (
    ReviewAction("a", "Accept raw", ACCEPT_RAW, "positive"),
    ReviewAction("c", "Accept candidate", ACCEPT_CANDIDATE, "positive"),
    ReviewAction("e", "Enter text", ENTER_TEXT, "warning", asks_for_text=True),
    ReviewAction("i", "Illegible", MARK_ILLEGIBLE, "negative"),
    ReviewAction("x", "Clear", CLEAR_DECISION, "neutral"),
)


class BookQueueAdapter(ReviewAdapter):
    """One uncertain span at a time. A decision is written to the run's overlay file."""

    kind = "book-review-queue"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.queue = read_queue(self.path)

    @property
    def actions(self) -> tuple[ReviewAction, ...]:
        return _ACTIONS

    def __len__(self) -> int:
        return len(self.queue.entries)

    def record(self, index: int) -> ReviewRecord:
        entry = self.queue.entries[index]
        return ReviewRecord(
            key=entry.entry_id,
            title=_title(entry),
            sections=(
                fields_section("Passage", _passage(entry), ("rendered", "marker"), "data"),
                fields_section(
                    "Readings",
                    _readings(entry),
                    ("raw", "candidate", "classes", "crop"),
                    "evidence",
                ),
                fields_section(
                    "Where",
                    _where(entry),
                    ("page", "block", "reason", "anchor", "earlier"),
                    "metadata",
                ),
            ),
            stratum=entry.kind,
            verdict=entry.verdict,
        )

    def apply(self, index: int, action: str) -> None:
        self.queue = apply_review_decision(self.path, index, action)


def _title(entry: ReviewQueueEntry) -> str:
    suffix = " (earlier decision orphaned)" if entry.priors else ""
    return f"{entry.kind} {entry.entry_id}{suffix}"


def _passage(entry: ReviewQueueEntry) -> dict[str, str]:
    return {"rendered": entry.rendered, "marker": entry.marker}


def _readings(entry: ReviewQueueEntry) -> dict[str, str]:
    classes = ", ".join(entry.classes)
    return {
        "raw": entry.raw_text,
        "candidate": entry.candidate or "",
        "classes": classes,
        "crop": entry.crop or "",
    }


def _where(entry: ReviewQueueEntry) -> dict[str, str]:
    earlier = "\n".join(
        f"orphaned ({prior.reason}): {prior.decision or 'decision'} -> {prior.replacement}"
        for prior in entry.priors
    )
    missing = "" if entry.block_present else " (block missing)"
    return {
        "page": f"PDF {entry.pdf_page} {entry.pdf_link}",
        "block": f"{entry.block_id} {entry.start}-{entry.end}{missing}",
        "reason": entry.reason,
        "anchor": entry.anchor,
        "earlier": earlier,
    }
