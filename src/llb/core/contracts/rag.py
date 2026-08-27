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
    # The encoder IDENTITY this store was built by (`llb.rag.encoders.tuned.embedder_fingerprint`).
    # For a hub id it repeats `embedding_model`; for a locally fine-tuned directory it is the base
    # model plus the tuned digest, because the directory path alone is not an identity -- training
    # again into it produces different weights under the same string. Optional so a store written
    # before the field existed still loads and still re-reads.
    embedder_fingerprint: NotRequired[str]
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
    collapse_duplicates: NotRequired[
        bool
    ]  # duplicate chunk collapse on/off (llb.rag.duplicates.collapse)
    duplicate_tier: NotRequired[str]  # when two texts are one passage (llb.rag.duplicates.tiers)
    duplicates: NotRequired[JsonObject]  # its measured DuplicateStats, collapsed or not


class RetrievalMetrics(TypedDict):
    n: int
    k: int
    recall_at_k: float
    mrr: float
    # Evidence INTACTNESS, the pair recall@k cannot see: recall fires on a one-character overlap,
    # so `span_char_coverage_at_k` reports how much of each gold span the top-k actually carries
    # and `span_intact_at_k` how often ONE chunk carries a span whole (see `llb.rag.retrieval`).
    # `evaluate_retrieval` always emits both; they are optional ONLY so a run manifest recorded
    # before the pair existed still validates and still re-reads.
    span_char_coverage_at_k: NotRequired[float]
    span_intact_at_k: NotRequired[float]
    # Mean served context characters of the top-k (`llb.rag.retrieval.served_chars_at_k`): the COST
    # beside the quality columns, so a lever that buys intactness by serving more text is told
    # apart from one that reflows the same characters. Optional on the same terms as the pair above.
    served_chars_at_k: NotRequired[float]


RetrievalPair: TypeAlias = tuple[list[ChunkRecord], list[SourceSpanRecord]]


class RetrievedOccurrence(TypedDict):
    """One other place a retrieved chunk's text appears (see `llb.rag.duplicates.collapse`)."""

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
    token_precision: float
    token_recall: float
    exact: float
    contains: float
    semantic: NotRequired[float]


class AnswerSpanScores(TypedDict):
    """Answer-side gold-span coverage of one case (`llb.scoring.answer_spans`).

    `answer_span_coverage` is the share of the item's judgeable gold spans whose fact the ANSWER
    states, and `answer_all_spans` its all-or-nothing gate -- the answer-side twins of
    `span_coverage_at_k` / `all_spans_at_k`, which say the same two things about the CONTEXT.
    `answer_spans_measured` counts the spans behind them, so a vacuous 1.0 (nothing judgeable) is
    never read as a carried one.
    """

    answer_span_coverage: float
    answer_all_spans: float
    answer_spans_measured: int


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
