"""Scanned-book recovery run contracts: the run configuration, the stage record, the page ledger.

A `book-ocr` run is resumable because these three records say everything a later invocation needs:
the configuration fixes what the run is (source identity, pages, stages, parameters, models), each
committed stage directory carries the record of the cache key it was produced under, and the page
manifest projects those records into one per-page, per-stage ledger.
"""

from typing import Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.corpus import DataPrepRow

SourcePolicy = Literal["copy", "symlink", "reference"]
StageStatus = Literal["done", "failed"]
PageStageStatus = Literal["done", "failed", "blocked", "pending"]
ImageRotation = Literal[0, 90, 180, 270]
Sha256Hex = str


class BookSourceRecord(DataPrepRow):
    """The source PDF's identity and how the run holds it."""

    original_path: str = Field(min_length=1)
    sha256: Sha256Hex = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=1)
    policy: SourcePolicy
    # Run-relative for copy/symlink, the absolute original for reference.
    stored_path: str = Field(min_length=1)


class BookStageSpec(DataPrepRow):
    """One configured stage: its registered name, its parameters, and the model it runs."""

    name: str = Field(min_length=1)
    params: JsonObject = Field(default_factory=dict)
    model: str | None = None


class BookOcrRunConfig(ArtifactContract):
    """`config.json`: what one `book-ocr` run is. A resume reads it instead of the command line."""

    schema_id: Literal["llb.book-ocr-run-config"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-run-config"]
    run_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    source: BookSourceRecord
    pages_spec: str = Field(min_length=1)
    pages: list[int] = Field(min_length=1)
    batch_pages: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    stages: list[BookStageSpec] = Field(min_length=1)
    reuse_from: str | None = None


class StageOutputRecord(DataPrepRow):
    """One file a committed stage wrote, relative to its stage directory."""

    path: str = Field(min_length=1)
    sha256: Sha256Hex = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class BookStageRecord(ArtifactContract):
    """`pages/<pNNNN>/<stage>/stage.json`: one stage's outcome on one page, and its cache key.

    The key is the digest of every component listed beside it, so a reader can see WHY two
    records share a key rather than trusting the digest alone.
    """

    schema_id: Literal["llb.book-ocr-stage-record"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-stage-record"]
    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    stage: str = Field(min_length=1)
    status: StageStatus
    cache_key: Sha256Hex = Field(pattern=r"^[0-9a-f]{64}$")
    stage_version: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    model: str | None = None
    params_digest: Sha256Hex = Field(pattern=r"^[0-9a-f]{64}$")
    input_digests: dict[str, str] = Field(default_factory=dict)
    attempts: int = Field(ge=1)
    duration_s: float = Field(ge=0)
    outputs: list[StageOutputRecord] = Field(default_factory=list)
    meta: JsonObject = Field(default_factory=dict)
    error: str | None = None
    reused_from: str | None = None


class PageInventoryRecord(DataPrepRow):
    """What the PDF says about one page before anything renders it."""

    width_pt: float = Field(gt=0)
    height_pt: float = Field(gt=0)
    rotation: int
    orientation: Literal["portrait", "landscape"]
    image_count: int = Field(ge=0)
    image_width_px: int | None = None
    image_height_px: int | None = None
    native_dpi: int | None = None
    # The quarter turn the largest image is PLACED at. It says nothing about whether the printed
    # content reads upright: that is a measured page signal, not a PDF property.
    image_rotation: ImageRotation | None = None
    embedded_text_chars: int = Field(ge=0)


class PageStageState(DataPrepRow):
    """One stage's state on one page, as the ledger projects it."""

    stage: str = Field(min_length=1)
    status: PageStageStatus
    cache_key: str | None = None
    attempts: int = Field(ge=0)
    duration_s: float = Field(ge=0)
    error: str | None = None
    reused_from: str | None = None


class PageLedgerRecord(DataPrepRow):
    """One page of the run: its identity, its inventory, its page-image digest, its stages."""

    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    batch: int = Field(ge=0)
    inventory: PageInventoryRecord
    image_sha256: str | None = None
    stages: list[PageStageState] = Field(default_factory=list)


class BookPageManifest(ArtifactContract):
    """`metadata/page_manifest.json`: every selected page and the state of every stage on it."""

    schema_id: Literal["llb.book-ocr-page-manifest"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-page-manifest"]
    run_id: str = Field(min_length=1)
    source_sha256: Sha256Hex = Field(pattern=r"^[0-9a-f]{64}$")
    source_page_count: int = Field(ge=1)
    n_pages: int = Field(ge=0)
    stage_names: list[str] = Field(default_factory=list)
    summary: dict[str, dict[str, int]] = Field(default_factory=dict)
    pages: list[PageLedgerRecord] = Field(default_factory=list)
