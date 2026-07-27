"""State and callback contracts for the single-call RAG graph."""

from typing import Callable

from typing_extensions import TypedDict

from llb.core.contracts.common import UsageRecord
from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord


class RagState(TypedDict, total=False):
    question: str
    gold_spans: list[SourceSpanRecord]
    retrieved: list[ChunkRecord]
    context: str
    answer: str
    status: str
    error: str | None
    usage: UsageRecord
    retrieve_latency_s: float
    rerank_latency_s: float
    query_processed: str
    query_corrections: int
    query_hypothetical_answer: str
    query_decomposition: str
    query_subqueries: list[str]


ContextSource = Callable[[RagState], RagState]
