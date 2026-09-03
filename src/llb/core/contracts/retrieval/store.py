"""Vector-store record contracts: the indexed chunk row and the store metadata document.

`chunks.jsonl`, `parents.jsonl`, and `store_meta.json` are what a retrieval run is served from,
and until now they were a `dict[str, object]` written with `json.dumps`. Registering them means a
store built by a newer writer is refused at the door instead of retrieving with a field this
reader cannot see, and that the rows can be read by anything that can read JSON -- no FAISS, no
embedding stack, no `llb` import.

The chunk `metadata` map stays open on purpose: it carries per-strategy span annotations, corpus
governance fields, page annotations, and duplicate-occurrence records, each owned by the module
that writes it. The contract states that the map is present, not what any producer may put in it.
"""

from typing import Literal

from pydantic import ConfigDict, Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject

RAG_CHUNK_SCHEMA_ID = "llb.rag-chunk"
RAG_STORE_META_SCHEMA_ID = "llb.rag-store-meta"


class RagChunkRecord(ArtifactContract):
    """One indexed unit: an offset-exact source span plus how it was chunked.

    A `parents.jsonl` row is the same record without `parent_id`; a `parent_child` child carries
    the parent it belongs to. Query-time fields (`retrieval_score`, `rank`, `rerank_score`) are
    never persisted -- they describe one answer, not the store.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.rag-chunk"]
    schema_version: Literal["1.0.0"]
    doc_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    strategy: str = Field(min_length=1)
    size: int = Field(ge=0)
    overlap: int | None = None
    parent_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class RagStoreMetaRecord(ArtifactContract):
    """`store_meta.json`: what a store is, and every knob a query applies to it.

    `backend`, `collapse_duplicates`, and `duplicate_tier` are optional because a store built
    before each knob existed recorded nothing, and stores in exactly that state are still on disk.
    The reader that needs one applies its documented default (`faiss`, collapse on, the `exact`
    tier); the contract says the store did not record it rather than inventing a value for it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.rag-store-meta"]
    schema_version: Literal["1.0.0"]
    mode: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    size: int = Field(ge=0)
    overlap: int = Field(ge=0)
    child_size: int = Field(ge=0)
    embedding_model: str = Field(min_length=1)
    n_indexed: int = Field(ge=0)
    n_parents: int = Field(ge=0)
    dim: int = Field(ge=0)
    backend: str | None = None
    collapse_duplicates: bool | None = None
    duplicate_tier: str | None = None
    embedder_fingerprint: str | None = None
    page_annotation_coverage: float | None = None
    corpus_fingerprint: str | None = None
    corpus_manifest: str | None = None
    refreshed_from: str | None = None
    governance_fields: list[str] | None = None
    doc_fingerprints: dict[str, str] | None = None
    lexical: JsonObject | None = None
    duplicates: JsonObject | None = None
