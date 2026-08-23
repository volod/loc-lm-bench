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
from llb.eval.answer_envelope import lane as envelope_lane
from llb.eval.answer_envelope import metrics as envelope_metrics
from llb.eval.answer_envelope.models import AnswerEnvelope
from llb.goldset.schema import GoldItem
from llb.rag import retrieval
from llb.rag.retrieval_records import retrieved_span
from llb.scoring import correctness, groundedness
from llb.scoring.verbosity import ranking_score

from llb.eval.graph_contracts import RagState


@dataclass(slots=True, frozen=True)
class ScoreOptions:
    """Opt-in answer-side scoring toggles (groundedness-citation-metrics).

    `context_order` mirrors the prompt-layout policy so citations are validated against the chunks
    in the exact order the model saw them, declared or scraped.

    `answer_format` says where the answer-side signals are READ FROM (typed-rag-answer-envelope):
    the declared `AnswerEnvelope` fields, or the prose heuristics. It does not decide WHICH signals
    are recorded -- the two toggles above still do that -- so an envelope run and a free-text run
    carry the same columns and stay comparable.
    """

    score_groundedness: bool = False
    cited_answers: bool = False
    context_order: str = eval_common.ORDER_RANK
    answer_format: str = envelope_lane.FREE_TEXT


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
        "token_precision": corr["token_precision"],
        "token_recall": corr["token_recall"],
        "ranking_score": ranking_score(corr["token_precision"], corr["token_recall"]),
        "exact": corr["exact"],
        "contains": corr["contains"],
        "retrieval_hit": retrieval.recall_at_k(retrieved, spans, len(retrieved)),
        "first_hit_rank": retrieval.first_hit_rank(retrieved, spans),
        "tokens_per_s": usage.get("tokens_per_s", 0.0),
        "latency_s": usage.get("latency_s", 0.0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "answer_preview": (answer or "")[:280],
    }
    if usage.get("prompt_tokens"):
        row["prompt_tokens"] = int(usage["prompt_tokens"])
    if "semantic" in corr:
        row["semantic"] = corr["semantic"]
    if "retrieve_latency_s" in state:
        row["retrieve_latency_s"] = round(float(state["retrieve_latency_s"]), 4)
    if "rerank_latency_s" in state:
        row["rerank_latency_s"] = round(float(state["rerank_latency_s"]), 4)
    if "query_processed" in state:
        row["query_processed"] = str(state["query_processed"])
        row["query_corrections"] = int(state.get("query_corrections", 0))
    if "query_dense" in state:
        row["query_dense"] = str(state["query_dense"])
    if "query_hypothetical_answer" in state:
        row["query_hypothetical_answer"] = str(state["query_hypothetical_answer"])
    if "query_decomposition" in state:
        row["query_decomposition"] = str(state["query_decomposition"])
    if "query_subqueries" in state:
        row["query_subqueries"] = [str(value) for value in state["query_subqueries"]]
    if "table_headers_restored" in state:
        row["table_headers_restored"] = int(state["table_headers_restored"])
        row["table_header_chars"] = float(state.get("table_header_chars", 0))
    envelope = _declared_envelope(state)
    _attach_envelope_columns(row, state, envelope)
    # Answer-side signals read the chunks as the PROMPT carried them, which is what the model was
    # asked to ground its answer in; they differ from `retrieved` only under prompt-side context
    # assembly (`llb.eval.table_headers`).
    _score_answer_side(row, answer, state.get("prompt_chunks") or retrieved, options, envelope)
    return row


def _attach_envelope_columns(
    row: CaseScoreRow, state: RagState, envelope: AnswerEnvelope | None
) -> None:
    """Attach the declared-answer columns (typed-rag-answer-envelope) to `row`.

    Present only on an envelope-format run, so every bundle recorded with the envelope off keeps
    exactly the shape it had. `envelope_status` is the parse verdict, `repaired` says the bounded
    reprompt was spent (which makes first-attempt conformance readable as `1 - repair_rate`), and
    `n_claims` / `envelope_abstained` are read straight off the declaration.
    """
    if "envelope_status" not in state:
        return
    row["envelope_status"] = str(state["envelope_status"])
    row["repaired"] = bool(state.get("envelope_repaired", False))
    row["n_claims"] = len(envelope.claims) if envelope is not None else 0
    row["envelope_abstained"] = bool(envelope.abstained) if envelope is not None else False


def _declared_envelope(state: RagState) -> AnswerEnvelope | None:
    """The validated envelope this case declared, if it produced one.

    The state carries it as a plain dict (the durability journal serializes state to JSON), so it
    is revalidated here through the same contract that admitted it at the generation boundary.
    """
    payload = state.get("envelope")
    return AnswerEnvelope.model_validate(payload) if payload is not None else None


def _score_answer_side(
    row: CaseScoreRow,
    answer: str,
    prompt_chunks: list[ChunkRecord],
    options: ScoreOptions | None,
    envelope: AnswerEnvelope | None = None,
) -> None:
    """Attach the opt-in answer-side signals (groundedness-citation-metrics) to `row`.

    Citations are validated against the chunks in prompt-layout order, so the numbering matches
    what `format_context` emitted to the model. When the case DECLARED an envelope, the claims and
    their citations are read from it instead of being scraped out of prose; the two scorers apply
    the same support threshold and countable-claim rule, so the columns stay comparable."""
    if options is None or not (options.score_groundedness or options.cited_answers):
        return
    ordered = eval_common.order_chunks(prompt_chunks, options.context_order)
    declared = envelope if options.answer_format == envelope_lane.ENVELOPE else None
    if options.score_groundedness:
        fraction = (
            envelope_metrics.envelope_groundedness(declared, ordered)
            if declared is not None
            else groundedness.groundedness_fraction(answer, ordered)
        )
        row["groundedness"] = round(fraction, 4)
    if options.cited_answers:
        report = (
            envelope_metrics.envelope_citation_report(declared, ordered)
            if declared is not None
            else groundedness.citation_report(answer, ordered)
        )
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
