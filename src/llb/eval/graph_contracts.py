"""State and callback contracts for the single-call RAG graph."""

from typing import Callable

from typing_extensions import TypedDict

from llb.core.contracts.common import JsonObject, UsageRecord
from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord


class RagState(TypedDict, total=False):
    question: str
    gold_spans: list[SourceSpanRecord]
    retrieved: list[ChunkRecord]
    # The chunks as the PROMPT carried them. Present only when prompt-side context assembly made
    # them differ from `retrieved` (table-header restoration); `retrieved` always stays the stored
    # records the source-span metrics are read from.
    prompt_chunks: list[ChunkRecord]
    context: str
    answer: str
    status: str
    error: str | None
    usage: UsageRecord
    retrieve_latency_s: float
    rerank_latency_s: float
    query_processed: str
    query_corrections: int
    query_dense: str
    query_hypothetical_answer: str
    query_decomposition: str
    query_subqueries: list[str]
    # The declared answer contract (typed-rag-answer-envelope), present only on an envelope-format
    # run. `envelope` is the VALIDATED envelope as a plain dict so the durability journal can carry
    # it; `envelope_status` is the parse verdict (ok / malformed / schema_invalid), `envelope_error`
    # the validator complaint behind a non-ok verdict, and `envelope_repaired` says whether the
    # bounded repair reprompt was spent on this case.
    envelope: JsonObject
    envelope_status: str
    envelope_error: str
    envelope_repaired: bool
    # Prompt-side table-header restoration accounting: how many retrieved chunks were given back
    # their column names, and what that added in characters (0 / 0 when the step is off).
    table_headers_restored: int
    table_header_chars: int


ContextSource = Callable[[RagState], RagState]
