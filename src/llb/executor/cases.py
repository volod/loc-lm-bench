"""Per-case evaluation execution and score-row construction."""

from dataclasses import dataclass
from typing import Any, Callable

from llb.core.contracts.rag import (
    CaseRetrievalRecord,
    ChunkRecord,
    RetrievalPair,
    SourceSpanRecord,
)
from llb.core.contracts.results import CaseScoreRow
from llb.eval import common as eval_common
from llb.eval import graph as eval_graph
from llb.goldset.schema import GoldItem
from llb.rag import retrieval
from llb.rag.retrieval_records import retrieved_span
from llb.scoring import correctness, groundedness

RagState = eval_graph.RagState


@dataclass(slots=True, frozen=True)
class ScoreOptions:
    """Opt-in answer-side scoring toggles (groundedness-citation-metrics).

    `context_order` mirrors the prompt-layout policy so `[i]` citations are validated against the
    chunks in the exact order the model saw them.
    """

    score_groundedness: bool = False
    cited_answers: bool = False
    context_order: str = eval_common.ORDER_RANK


@dataclass(slots=True)
class CaseBatch:
    """Outputs collected while evaluating a batch of gold items."""

    rows: list[CaseScoreRow]
    retrieval_pairs: list[RetrievalPair]
    answers: list[tuple[GoldItem, str]]


def spans_as_dicts(item: GoldItem) -> list[SourceSpanRecord]:
    return [
        {
            "doc_id": span.doc_id,
            "char_start": span.char_start,
            "char_end": span.char_end,
            "text": span.text,
        }
        for span in item.source_spans
    ]


def batch_retrieval_records(batch: "CaseBatch") -> list[CaseRetrievalRecord]:
    """The per-case retrieved-spans records persisted as `retrieval.jsonl` (miss analysis):
    what each case's context actually contained versus its gold spans.

    The item's gold spans are passed into each record so a collapsed chunk keeps the occurrences
    that decide ITS metric, and a lane recomputing from the sidecar agrees with the run
    (`llb.rag.retrieval_records`)."""
    return [
        {
            "item_id": item.id,
            "retrieved": [
                retrieved_span(chunk, rank, spans) for rank, chunk in enumerate(retrieved, 1)
            ],
            "gold_spans": spans,
        }
        for (item, _answer), (retrieved, spans) in zip(batch.answers, batch.retrieval_pairs)
    ]


def score_case(
    item: GoldItem,
    state: RagState,
    embedder: Any = None,
    options: ScoreOptions | None = None,
) -> CaseScoreRow:
    """Build one per-case score row from a terminal graph state."""
    answer = state.get("answer", "")
    status = state.get("status", eval_common.OK)
    spans = spans_as_dicts(item)
    retrieved = state.get("retrieved", [])
    corr = correctness.answer_correctness(answer, item.reference_answer, embedder=embedder)
    usage = state.get("usage", {})
    row: CaseScoreRow = {
        "item_id": item.id,
        "split": item.split,
        "status": status,
        "objective_score": corr["score"],
        "token_f1": corr["token_f1"],
        "exact": corr["exact"],
        "contains": corr["contains"],
        "retrieval_hit": retrieval.recall_at_k(retrieved, spans, len(retrieved)),
        "first_hit_rank": retrieval.first_hit_rank(retrieved, spans),
        "tokens_per_s": usage.get("tokens_per_s", 0.0),
        "latency_s": usage.get("latency_s", 0.0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "answer_preview": (answer or "")[:280],
    }
    if "semantic" in corr:
        row["semantic"] = corr["semantic"]
    if "retrieve_latency_s" in state:
        row["retrieve_latency_s"] = round(float(state["retrieve_latency_s"]), 4)
    if "rerank_latency_s" in state:
        row["rerank_latency_s"] = round(float(state["rerank_latency_s"]), 4)
    if "query_processed" in state:
        row["query_processed"] = str(state["query_processed"])
        row["query_corrections"] = int(state.get("query_corrections", 0))
    if "query_hypothetical_answer" in state:
        row["query_hypothetical_answer"] = str(state["query_hypothetical_answer"])
    if "query_decomposition" in state:
        row["query_decomposition"] = str(state["query_decomposition"])
    if "query_subqueries" in state:
        row["query_subqueries"] = [str(value) for value in state["query_subqueries"]]
    _score_answer_side(row, answer, retrieved, options)
    return row


def _score_answer_side(
    row: CaseScoreRow,
    answer: str,
    retrieved: list[ChunkRecord],
    options: ScoreOptions | None,
) -> None:
    """Attach the opt-in answer-side signals (groundedness-citation-metrics) to `row`.

    `[i]` citations are validated against the chunks in prompt-layout order, so the numbering
    matches what `format_context` emitted to the model."""
    if options is None or not (options.score_groundedness or options.cited_answers):
        return
    ordered = eval_common.order_chunks(retrieved, options.context_order)
    if options.score_groundedness:
        row["groundedness"] = round(groundedness.groundedness_fraction(answer, ordered), 4)
    if options.cited_answers:
        report = groundedness.citation_report(answer, ordered)
        row["citation_validity"] = round(report["citation_validity"], 4)
        row["citation_coverage"] = round(report["citation_coverage"], 4)
        row["hallucinated_citation_rate"] = round(report["hallucinated_citation_rate"], 4)
        row["n_citations"] = report["n_citations"]


def execute_cases(
    items: list[GoldItem],
    runner_fn: Callable[[GoldItem], RagState],
    embedder: Any,
    options: ScoreOptions | None = None,
) -> CaseBatch:
    """Evaluate all items sequentially and collect scoring, retrieval, and answer outputs."""
    rows: list[CaseScoreRow] = []
    retrieval_pairs: list[RetrievalPair] = []
    answers: list[tuple[GoldItem, str]] = []
    for item in items:
        state = runner_fn(item)
        spans = spans_as_dicts(item)
        rows.append(score_case(item, state, embedder=embedder, options=options))
        retrieval_pairs.append((state.get("retrieved", []), spans))
        answers.append((item, state.get("answer", "")))
    return CaseBatch(rows=rows, retrieval_pairs=retrieval_pairs, answers=answers)
