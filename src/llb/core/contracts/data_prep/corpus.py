"""Corpus and PDF ingestion record contracts.

The manifests here are what every later stage keys on: a store build reads the corpus manifest to
fingerprint a generation, drafting reads it for per-document governance, and span validation reads
the citation sidecar to cite an originating PDF page. Each is a strict model rather than the
`dict[str, object]` the writers used to hand to `json.dumps`, so a member that lost a field is
refused where it is read instead of surfacing as a missing key much later.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject


class DataPrepRow(BaseModel):
    """Strict nested row shared by the data-prep manifests."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CorpusItemRecord(DataPrepRow):
    """One source document's ingestion outcome, mirroring `llb.prep.corpus.CorpusItem`.

    Every governance field is present on every row, `None` where the corpus carries none, so a
    manifest states an absence rather than leaving a reader to infer one.
    """

    source: str = Field(min_length=1)
    doc_id: str | None
    kind: Literal["pdf", "text"]
    status: Literal["ok", "too_short", "error"]
    n_chars: int = Field(ge=0)
    source_sha256: str | None = None
    reused: bool = False
    error: str | None = None
    parser: str | None = None
    language: str | None = None
    version: str | None = None
    effective_date: str | None = None
    ingestion_time: str | None = None
    source_system: str | None = None
    acl_label: str | None = None
    source_uri: str | None = None
    capture_time: str | None = None
    capture_id: str | None = None
    payload_digest: str | None = None
    licence: str | None = None
    acquisition_run_id: str | None = None
    revision_of: str | None = None


class CorpusManifest(ArtifactContract):
    """`corpus_manifest.json`: every ingested source, its status, and the corpus fingerprint."""

    schema_id: Literal["llb.corpus-manifest"]
    schema_version: Literal["1.0.0"]
    kind: Literal["corpus"]
    source_root: str = Field(min_length=1)
    corpus_root: str = Field(min_length=1)
    n_sources: int = Field(ge=0)
    n_docs: int = Field(ge=0)
    n_skipped: int = Field(ge=0)
    n_reused: int = Field(ge=0)
    n_removed_sources: int = Field(ge=0)
    removed_sources: list[str] = Field(default_factory=list)
    corpus_fingerprint: str | None = None
    governance_coverage: JsonObject = Field(default_factory=dict)
    items: list[CorpusItemRecord] = Field(default_factory=list)


class PdfItemRecord(DataPrepRow):
    """One PDF conversion outcome, mirroring `llb.prep.pdf.model.PdfCorpusItem`."""

    source: str = Field(min_length=1)
    doc_id: str | None
    n_chars: int = Field(ge=0)
    status: str = Field(min_length=1)
    error: str | None = None
    parser: str | None = None
    citation_path: str | None = None
    page_count: int | None = None
    embedded_text_chars: int | None = None
    image_only_pages: int | None = None
    attempts: list[JsonObject] = Field(default_factory=list)
    diagnostics: JsonObject | None = None
    quality: JsonObject | None = None
    source_sha256: str | None = None
    reused: bool = False
    repeat_blocks: str = Field(default="", min_length=0)


class PdfCorpusManifest(ArtifactContract):
    """`pdf_corpus_manifest.json` and `pdf_corpus_quality.json`; `kind` separates the two."""

    schema_id: Literal["llb.pdf-corpus-manifest"]
    schema_version: Literal["1.0.0"]
    kind: Literal["pdf-corpus", "pdf-corpus-quality"]
    pdf_root: str = Field(min_length=1)
    corpus_root: str = Field(min_length=1)
    n_pdfs: int = Field(ge=0)
    n_docs: int = Field(ge=0)
    n_skipped: int = Field(ge=0)
    items: list[PdfItemRecord] = Field(default_factory=list)


class PdfPageCitationRecord(DataPrepRow):
    """One rendered page's char offsets into the staged markdown."""

    page: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text_start: int = Field(ge=0)
    text_end: int = Field(ge=0)
    n_chars: int = Field(ge=0)
    parser: str = Field(min_length=1)
    blocks: list[JsonObject] = Field(default_factory=list)


class PdfCitations(ArtifactContract):
    """`<doc>.citations.json`: the page a staged character offset came from."""

    schema_id: Literal["llb.pdf-citations"]
    schema_version: Literal["1.0.0"]
    kind: Literal["pdf-citations"]
    source: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    parser: str = Field(min_length=1)
    diagnostics: JsonObject = Field(default_factory=dict)
    pages: list[PdfPageCitationRecord] = Field(default_factory=list)
