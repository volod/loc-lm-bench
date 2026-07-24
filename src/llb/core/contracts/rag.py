"""Retrieval, corpus, and RAG dataset contracts."""

from typing import TypeAlias

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.common import JsonObject


class SourceSpanRecord(TypedDict):
    doc_id: str
    char_start: int
    char_end: int
    text: str


class ChunkRecord(SourceSpanRecord):
    chunk_id: NotRequired[str]
    parent_id: NotRequired[str]
    matched_child_id: NotRequired[str]
    strategy: NotRequired[str]
    size: NotRequired[int]
    overlap: NotRequired[int]
    metadata: NotRequired[JsonObject]
    retrieval_score: NotRequired[float | None]
    rank: NotRequired[int]
    rerank_score: NotRequired[float]
    pre_rerank_rank: NotRequired[int]


class RagStoreMeta(TypedDict):
    mode: str
    strategy: str
    size: int
    overlap: int
    child_size: int
    embedding_model: str
    n_indexed: int
    n_parents: int
    dim: int
    backend: NotRequired[str]
    page_annotation_coverage: NotRequired[float]
    lexical: NotRequired[JsonObject]
    corpus_fingerprint: NotRequired[str]
    corpus_manifest: NotRequired[str]
    governance_fields: NotRequired[list[str]]
    doc_fingerprints: NotRequired[dict[str, str]]
    refreshed_from: NotRequired[str]
    collapse_duplicates: NotRequired[bool]  # duplicate chunk collapse on/off (llb.rag.duplicates)
    duplicate_tier: NotRequired[str]  # when two texts are one passage (llb.rag.duplicate_tiers)
    duplicates: NotRequired[JsonObject]  # its measured DuplicateStats, collapsed or not


class RetrievalMetrics(TypedDict):
    n: int
    k: int
    recall_at_k: float
    mrr: float


RetrievalPair: TypeAlias = tuple[list[ChunkRecord], list[SourceSpanRecord]]


class RetrievedOccurrence(TypedDict):
    """One other place a retrieved chunk's text appears (see `llb.rag.duplicates`)."""

    doc_id: str
    char_start: int
    char_end: int
    chunk_id: NotRequired[str]


class RetrievedSpanRecord(TypedDict):
    """Bounded retrieved-span data persisted for miss analysis."""

    doc_id: str
    char_start: int
    char_end: int
    rank: int
    retrieval_score: NotRequired[float | None]
    text_preview: NotRequired[str]
    # Present only for a chunk that collapsed byte-identical copies: the TOTAL number of places
    # its text appears (including this one), and a bounded, gold-complete list of the others --
    # see `llb.rag.retrieval_records`. An uncollapsed chunk carries neither key.
    duplicate_count: NotRequired[int]
    duplicate_occurrences: NotRequired[list[RetrievedOccurrence]]


class CaseRetrievalRecord(TypedDict):
    """Retrieved and gold spans persisted for one scored case."""

    item_id: str
    retrieved: list[RetrievedSpanRecord]
    gold_spans: list[SourceSpanRecord]


class CorrectnessScores(TypedDict):
    score: float
    token_f1: float
    exact: float
    contains: float
    semantic: NotRequired[float]


class ChunkSummary(TypedDict):
    """Chunk-length distribution of a built store; the oversize fields audit the `size` cap."""

    n: int
    avg: int
    min: int
    max: int
    oversize: int  # chunks longer than the `size` they were built with
    oversize_share: float  # their share of the chunk COUNT
    oversize_char_share: float  # their share of the indexed CHARACTERS


class SquadAnswers(TypedDict):
    text: list[str]
    answer_start: list[int | None]


class SquadRecord(TypedDict):
    id: str | None
    context: str
    question: str
    answers: SquadAnswers


class RagItemSpec(TypedDict):
    id: str
    doc: str
    answer_span: str
    question: str
    reference_answer: str
    split: str
    provenance: NotRequired[str]
    verified: NotRequired[bool]


class RagDataSpec(TypedDict):
    lang: str
    docs: dict[str, str]
    items: list[RagItemSpec]
