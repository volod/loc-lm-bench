"""Review-queue and page-quality records for one scanned book.

A reader sees an uncertainty marker instead of a silent gap. The queue is the ledger the review
workbench opens. The quality manifest is the page-level list of those same spans. Reviewer
decisions themselves stay in `review/decisions.json`; this file only records that they were made.
"""

from typing import Final, Literal

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.data_prep.book_blocks import UncertaintyClass
from llb.core.contracts.data_prep.book_resolve import OrphanReason
from llb.core.contracts.data_prep.corpus import DataPrepRow

MarkerKind = Literal["unreadable", "low-confidence", "formula-table"]
MARKER_KINDS: tuple[MarkerKind, ...] = ("unreadable", "low-confidence", "formula-table")
QueueReason = Literal["unresolved", "empty", "low-confidence", "orphaned"]
QUEUE_REASONS: tuple[QueueReason, ...] = ("unresolved", "empty", "low-confidence", "orphaned")
ReviewVerdict = Literal["", "accept-raw", "accept-candidate", "enter-text", "mark-illegible"]
REVIEW_VERDICTS: tuple[str, ...] = (
    "accept-raw",
    "accept-candidate",
    "enter-text",
    "mark-illegible",
)

QUEUE_SCHEMA_ID: Final = "llb.book-ocr-review-queue"
QUEUE_KIND: Final = "book-ocr-review-queue"
MANIFEST_SCHEMA_ID: Final = "llb.book-ocr-quality-manifest"
MANIFEST_KIND: Final = "book-ocr-quality-manifest"
REVIEW_SCHEMA_VERSION: Final = "1.0.0"
_ANCHOR = r"^[0-9a-f]{64}$"


class PriorDecision(DataPrepRow):
    """A decision the current ledger refused, shown again so it is not lost."""

    edit_id: str = Field(min_length=1)
    decision: str = ""
    replacement: str
    evidence: str = Field(min_length=1)
    reason: OrphanReason


class ReviewQueueEntry(DataPrepRow):
    """One span a reviewer can accept, replace, or mark illegible."""

    entry_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    block_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    anchor: str = Field(pattern=_ANCHOR)
    kind: MarkerKind
    reason: QueueReason
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    raw_text: str
    candidate: str | None = None
    classes: list[UncertaintyClass] = Field(default_factory=list)
    marker: str = Field(min_length=1)
    crop: str | None = None
    pdf_link: str = Field(min_length=1)
    rendered: str = Field(min_length=1)
    priors: list[PriorDecision] = Field(default_factory=list)
    verdict: ReviewVerdict = ""
    block_present: bool = True

    @model_validator(mode="after")
    def _span_and_marker_agree(self) -> "ReviewQueueEntry":
        if self.end < self.start:
            raise ValueError("queue entry end is before its start")
        if len(self.raw_text) != self.end - self.start:
            raise ValueError("queue entry raw text length does not match its offsets")
        if "#page=" not in self.pdf_link:
            raise ValueError("queue entry link must name a PDF page")
        if not self.marker.startswith("[!"):
            raise ValueError("queue entry marker must be a visible uncertainty marker")
        if not self.rendered.strip():
            raise ValueError("queue entry must show its marker, not a blank passage")
        return self


class ReviewQueue(ArtifactContract):
    """`review/queue.json`: every open uncertainty, in reading order."""

    schema_id: Literal["llb.book-ocr-review-queue"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-review-queue"]
    source_sha256: str = Field(pattern=_ANCHOR)
    entries: list[ReviewQueueEntry] = Field(default_factory=list)


class QualityRow(DataPrepRow):
    """One uncertain span on one page, the same span the queue carries."""

    entry_id: str = Field(min_length=1)
    block_id: str = Field(min_length=1)
    kind: MarkerKind
    reason: QueueReason
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    classes: list[UncertaintyClass] = Field(default_factory=list)
    marker: str = Field(min_length=1)
    crop: str | None = None
    raw_text: str


class QualityPage(DataPrepRow):
    """Every quality row on one PDF page. A page with no rows was still checked."""

    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    rows: list[QualityRow] = Field(default_factory=list)


class QualityManifest(ArtifactContract):
    """`metadata/quality_manifest.json`: one row per uncertain span, grouped by page."""

    schema_id: Literal["llb.book-ocr-quality-manifest"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-quality-manifest"]
    source_sha256: str = Field(pattern=_ANCHOR)
    counts: dict[str, int] = Field(default_factory=dict)
    pages: list[QualityPage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _counts_match_rows(self) -> "QualityManifest":
        found: dict[str, int] = {}
        for page in self.pages:
            for row in page.rows:
                found[row.kind] = found.get(row.kind, 0) + 1
        if self.counts != found:
            raise ValueError("quality manifest counts must match its rows")
        return self
