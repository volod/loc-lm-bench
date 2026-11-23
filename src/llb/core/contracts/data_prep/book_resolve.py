"""Resolved page text and the correction overlay that produced it.

The block ledger stays the raw record. A rendering reads the resolved page, whose every span says
whether it came from the configured primary reading, from an overlay edit, or from an uncertainty
span that no edit covered. The overlay is the only place a correction is stored.
"""

from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.data_prep.book_blocks import UncertaintyClass
from llb.core.contracts.data_prep.corpus import DataPrepRow

AuthorKind = Literal["agreement", "model", "reviewer"]
AUTHOR_KINDS: tuple[AuthorKind, ...] = ("reviewer", "model", "agreement")
OrphanReason = Literal["block_missing", "block_changed", "raw_mismatch"]
ORPHAN_REASONS: tuple[OrphanReason, ...] = ("block_missing", "block_changed", "raw_mismatch")
Provenance = Literal["primary", "agreement", "model", "reviewer", "unresolved"]
PROVENANCE: tuple[Provenance, ...] = ("primary", "agreement", "model", "reviewer", "unresolved")
_ANCHOR = r"^[0-9a-f]{64}$"
_OVERLAY_AUTHORS: frozenset[str] = frozenset(("agreement", "model", "reviewer"))


class CorrectionEdit(DataPrepRow):
    """One replacement anchored to the block that was read, not only to its id.

    Offsets are into the primary reading named when the edit was made. `anchor` is the digest of
    that block's region and raw readings, so a later ledger that reused the id for a different
    block cannot take the edit.
    """

    edit_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    block_id: str = Field(min_length=1)
    anchor: str = Field(pattern=_ANCHOR)
    author: AuthorKind
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    raw_text: str
    replacement: str
    evidence: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _raw_matches_the_span(self) -> "CorrectionEdit":
        if self.end < self.start:
            raise ValueError("correction end is before its start")
        if len(self.raw_text) != self.end - self.start:
            raise ValueError("correction raw text length does not match its offsets")
        return self


class OrphanedEdit(DataPrepRow):
    """An edit this ledger refused. It is re-queued and never applied to a different block."""

    edit: CorrectionEdit
    reason: OrphanReason
    requeued: Literal[True]


class PageCorrectionOverlay(ArtifactContract):
    """`pages/<pNNNN>/overlay.json`: edits applied on this page, and edits the ledger refused."""

    schema_id: Literal["llb.book-ocr-correction-overlay"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-correction-overlay"]
    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    applied: list[CorrectionEdit] = Field(default_factory=list)
    orphaned: list[OrphanedEdit] = Field(default_factory=list)


class ResolvedSpan(DataPrepRow):
    """One slice of the primary reading after resolution.

    `start` and `end` are offsets into the primary reading. An overlay replacement may be a
    different length than the slice it covers. An unresolved span keeps the primary characters
    and is not part of the adopted text.
    """

    provenance: Provenance
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    adopted: bool
    edit_id: str | None = None
    classes: list[UncertaintyClass] = Field(default_factory=list)

    @model_validator(mode="after")
    def _provenance_matches_the_span(self) -> "ResolvedSpan":
        if self.end < self.start:
            raise ValueError("resolved span end is before its start")
        if self.provenance == "unresolved":
            if self.adopted or not self.classes or self.edit_id is not None:
                raise ValueError("an unresolved span is unadopted, classed, and has no edit")
        elif not self.adopted or self.classes:
            raise ValueError("a resolved span is adopted and carries no uncertainty class")
        needs_edit = self.provenance in _OVERLAY_AUTHORS
        if needs_edit == (self.edit_id is None):
            raise ValueError("an overlay span names its edit, and a primary span does not")
        if self.provenance in ("primary", "agreement", "unresolved"):
            if len(self.text) != self.end - self.start:
                raise ValueError("this span's text must match the primary slice")
        return self


class ResolvedBlock(DataPrepRow):
    """One block's adopted text, the reading it was resolved from, and every span."""

    block_id: str = Field(min_length=1)
    anchor: str = Field(pattern=_ANCHOR)
    engine: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    model: str | None = None
    fallback: bool
    primary_text: str
    text: str
    spans: list[ResolvedSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def _spans_tile_the_primary(self) -> "ResolvedBlock":
        cursor = 0
        adopted: list[str] = []
        for span in self.spans:
            if span.start != cursor:
                raise ValueError("resolved spans must tile the primary reading")
            cursor = span.end
            if span.adopted:
                adopted.append(span.text)
        if cursor != len(self.primary_text):
            raise ValueError("resolved spans must cover the primary reading")
        if self.text != "".join(adopted):
            raise ValueError("resolved text must equal the adopted spans")
        return self


class PageResolvedText(ArtifactContract):
    """`pages/<pNNNN>/resolved.json`: the text both renderings read for one page."""

    schema_id: Literal["llb.book-ocr-resolved-page"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-resolved-page"]
    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    page_class: str
    primary_stage: str = Field(min_length=1)
    primary_engine: str = Field(min_length=1)
    primary_variant: str = Field(min_length=1)
    primary_model: str | None = None
    blocks: list[ResolvedBlock] = Field(default_factory=list)


class ModelCorrections(ArtifactContract):
    """`pages/<pNNNN>/correct/corrections.json`: edits a correction stage proposes.

    Resolve reads this file. It does not produce it.
    """

    schema_id: Literal["llb.book-ocr-model-corrections"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-model-corrections"]
    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    edits: list[CorrectionEdit] = Field(default_factory=list)

    @model_validator(mode="after")
    def _edits_belong_to_this_page(self) -> "ModelCorrections":
        for edit in self.edits:
            if edit.author != "model":
                raise ValueError("a model correction must be authored by the model")
            if edit.page_id != self.page_id:
                raise ValueError("a model correction must name its own page")
        return self


class ReviewerDecisions(ArtifactContract):
    """`review/decisions.json`: reviewer edits for the whole run, outside the stage cache."""

    schema_id: Literal["llb.book-ocr-reviewer-decisions"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-reviewer-decisions"]
    edits: list[CorrectionEdit] = Field(default_factory=list)

    @model_validator(mode="after")
    def _authors_are_reviewers(self) -> "ReviewerDecisions":
        for edit in self.edits:
            if edit.author != "reviewer":
                raise ValueError("a reviewer decision must be authored by the reviewer")
        return self
