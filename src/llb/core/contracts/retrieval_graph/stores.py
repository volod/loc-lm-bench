"""Vector-store generation contracts: the indexed rows and the store's own metadata.

A built store is a SET of files that only mean something together -- the chunk rows carry the
source-span coordinates every retrieval metric is scored on, the parent rows carry the generation
context a child hit stands for, and the metadata says which encoder produced the vectors that both
are ranked by. The rows are project-owned records and are modelled here; the vector index and the
lexical posting list are not, and are named as opaque members instead (`OpaqueIndexMember`).
"""

from typing import Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.retrieval_graph.common import OpaqueIndexMember, RetrievalRow

CHUNK_SCHEMA_ID = "llb.rag-chunk"
STORE_META_SCHEMA_ID = "llb.rag-store-meta"


class ChunkRow(ArtifactContract):
    """One indexed unit or one parent: `chunks.jsonl` and `parents.jsonl` share this contract.

    The two files hold the same record at different granularities -- a parent IS a chunk that was
    not indexed -- so one family binds both, and a store's manifest distinguishes them by member
    rather than by shape. The retrieval-time fields (`retrieval_score`, `rank`, `rerank_score`,
    `pre_rerank_rank`, `matched_child_id`) belong to a returned hit and are absent on disk; they
    are declared because the same record is what a lane hands to scoring.
    """

    schema_id: Literal["llb.rag-chunk"]
    schema_version: Literal["1.0.0"]
    doc_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    chunk_id: str | None = None
    parent_id: str | None = None
    matched_child_id: str | None = None
    strategy: str | None = None
    size: int | None = Field(default=None, ge=0)
    overlap: int | None = Field(default=None, ge=0)
    # Chunker-, page-, and duplicate-collapse metadata. It stays a JSON object because its keys
    # are owned by the chunking strategy that emitted them (`llb.rag.chunking.table` shifts span
    # metadata, `llb.rag.duplicates.collapse` adds occurrences), and a store must round-trip a
    # strategy's own keys rather than the set one release happened to know.
    metadata: JsonObject | None = None
    retrieval_score: float | None = None
    rank: int | None = Field(default=None, ge=0)
    rerank_score: float | None = None
    pre_rerank_rank: int | None = Field(default=None, ge=0)


class DuplicateCensus(RetrievalRow):
    """The duplicate rate of a chunk set at one tier (`llb.rag.duplicates.collapse`)."""

    tier: str | None = None  # absent in a store meta written before the tiers shipped (== exact)
    n: int = Field(ge=0)
    unique: int = Field(ge=0)
    collapsed: int = Field(ge=0)
    duplicate_chunks: int = Field(ge=0)
    duplicate_share: float
    groups: int = Field(ge=0)
    largest_group: int = Field(ge=0)
    intra_document_groups: int = Field(ge=0)
    cross_document_groups: int = Field(ge=0)


class LexicalSummary(RetrievalRow):
    """What the BM25 side of a hybrid store holds, stated without loading its postings."""

    lemmatize: bool
    n_terms: int = Field(ge=0)


class RagStoreMetaDocumentV1(ArtifactContract):
    """`store_meta.json` as written before a store recorded its opaque index members.

    The duplicate-collapse knobs are optional here for one reason: this single version covers
    every store this project wrote before the registry existed, and those span the release where
    collapse shipped. A meta that states them is carried through; one that does not has them
    supplied by the migration from the same constants its readers were already defaulting to.
    """

    schema_id: Literal["llb.rag-store-meta"]
    schema_version: Literal["1.0.0"]
    mode: str
    strategy: str
    size: int = Field(ge=0)
    overlap: int = Field(ge=0)
    child_size: int = Field(ge=0)
    embedding_model: str
    # The encoder IDENTITY the store was built by (`llb.rag.encoders.tuned.embedder_fingerprint`).
    # Absent in a store written before the field existed, where the model id is all there is.
    embedder_fingerprint: str | None = None
    n_indexed: int = Field(ge=0)
    n_parents: int = Field(ge=0)
    dim: int = Field(ge=0)
    backend: str | None = None
    page_annotation_coverage: float | None = None
    lexical: LexicalSummary | None = None
    corpus_fingerprint: str | None = None
    corpus_manifest: str | None = None
    governance_fields: list[str] | None = None
    doc_fingerprints: dict[str, str] | None = None
    refreshed_from: str | None = None
    collapse_duplicates: bool | None = None
    duplicate_tier: str | None = None
    duplicates: DuplicateCensus | None = None


class RagStoreMetaDocument(ArtifactContract):
    """`store_meta.json`: what a store was built from, by which encoder, over which corpus.

    Everything a caller must agree with before it may query the store lives here, which is why
    this is the member the load-time gate reads first: a mismatched encoder identity, a moved
    corpus fingerprint, or an index member whose bytes changed are all decided from this record
    alone, before an embedding stack or a vector backend is imported.
    """

    schema_id: Literal["llb.rag-store-meta"]
    schema_version: Literal["2.0.0"]
    mode: str
    strategy: str
    size: int = Field(ge=0)
    overlap: int = Field(ge=0)
    child_size: int = Field(ge=0)
    embedding_model: str
    embedder_fingerprint: str | None = None
    n_indexed: int = Field(ge=0)
    n_parents: int = Field(ge=0)
    dim: int = Field(ge=0)
    backend: str | None = None
    page_annotation_coverage: float | None = None
    lexical: LexicalSummary | None = None
    corpus_fingerprint: str | None = None
    corpus_manifest: str | None = None
    governance_fields: list[str] | None = None
    doc_fingerprints: dict[str, str] | None = None
    refreshed_from: str | None = None
    # Stated rather than left to a reader's constant: a store that kept its duplicates and one
    # that collapsed them index different units, and an incremental refresh reads the answer here.
    collapse_duplicates: bool
    duplicate_tier: str
    duplicates: DuplicateCensus | None = None
    # The files this store cannot be queried without whose format belongs to another library.
    # Empty in a generation migrated from version 1; a reader treats that as "this generation does
    # not state its index members", never as "it has none".
    index_members: list[OpaqueIndexMember] = Field(default_factory=list)
