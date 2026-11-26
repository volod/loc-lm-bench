"""Portable pilot-review inputs, per-page revisions, and metadata-only acceptance."""

from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.book_blocks import BlockKind
from llb.core.contracts.data_prep.corpus import DataPrepRow

PILOT_REVIEW_MANIFEST: Final = "llb.book-ocr-pilot-review"
PILOT_PAGE_REVISION: Final = "llb.book-ocr-pilot-revision"
PILOT_REFERENCE_ACCEPTANCE: Final = "llb.book-ocr-pilot-acceptance"
PILOT_REVIEW_VERSION: Final = "1.0.0"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class AlternateReading(DataPrepRow):
    kind: BlockKind
    text: str
    region: dict[Literal["x0", "y0", "x1", "y1"], float]


class DraftNote(DataPrepRow):
    block_id: str
    flags: list[str] = Field(min_length=1)
    alternate: list[AlternateReading]


class PilotScanPage(DataPrepRow):
    id: str = Field(pattern=r"^p[0-9]+$")
    number: int = Field(ge=1)
    page_class: str
    orientation: int
    scan_sha256: Sha256
    notes: list[DraftNote]


class PilotReviewManifest(ArtifactContract):
    """The frozen scan, score-format pre-draft, and blind notes for the selected pages."""

    schema_id: Literal["llb.book-ocr-pilot-review"]
    schema_version: Literal["1.0.0"]
    source_sha256: Sha256
    pilot_sha256: Sha256
    predraft_sha256: Sha256
    pages: list[PilotScanPage] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_pages(self) -> Self:
        if len({p.id for p in self.pages}) != len(self.pages):
            raise ValueError("review page IDs repeat")
        if len({p.number for p in self.pages}) != len(self.pages):
            raise ValueError("review PDF page numbers repeat")
        return self


class PilotPageRevision(ArtifactContract):
    """An append-only edit in the original scan frame, optionally accepted."""

    schema_id: Literal["llb.book-ocr-pilot-revision"]
    schema_version: Literal["1.0.0"]
    page: JsonObject = Field(description="Score-format page draft, with optional incomplete edits.")
    reviewer: str
    accepted: bool
    saved_at: str | None


class PilotPageAcceptance(DataPrepRow):
    pdf_page: int = Field(ge=1)
    reviewer: str = Field(min_length=1)
    accepted_at: str = Field(min_length=1)
    scan_sha256: Sha256
    page_sha256: Sha256
    revision: int = Field(ge=1)


class PilotReferenceAcceptance(ArtifactContract):
    """The hashes and decisions that bind a score input to the reviewed scan and frozen pilot."""

    schema_id: Literal["llb.book-ocr-pilot-acceptance"]
    schema_version: Literal["1.0.0"]
    source_sha256: Sha256
    pilot_sha256: Sha256
    predraft_sha256: Sha256
    reference_sha256: Sha256
    pages: list[PilotPageAcceptance] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_pages(self) -> Self:
        if len({p.pdf_page for p in self.pages}) != len(self.pages):
            raise ValueError("accepted PDF page numbers repeat")
        return self
