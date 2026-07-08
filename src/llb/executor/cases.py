"""Per-case evaluation execution and score-row construction."""

from dataclasses import dataclass
from typing import Any, Callable

from llb.core.contracts import CaseScoreRow, RetrievalPair, SourceSpanRecord
from llb.eval import common as eval_common
from llb.eval import graph as eval_graph
from llb.goldset.schema import GoldItem
from llb.rag import retrieval
from llb.scoring import correctness

RagState = eval_graph.RagState


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


def score_case(item: GoldItem, state: RagState, embedder: Any = None) -> CaseScoreRow:
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
    return row


def execute_cases(
    items: list[GoldItem],
    runner_fn: Callable[[GoldItem], RagState],
    embedder: Any,
) -> CaseBatch:
    """Evaluate all items sequentially and collect scoring, retrieval, and answer outputs."""
    rows: list[CaseScoreRow] = []
    retrieval_pairs: list[RetrievalPair] = []
    answers: list[tuple[GoldItem, str]] = []
    for item in items:
        state = runner_fn(item)
        spans = spans_as_dicts(item)
        rows.append(score_case(item, state, embedder=embedder))
        retrieval_pairs.append((state.get("retrieved", []), spans))
        answers.append((item, state.get("answer", "")))
    return CaseBatch(rows=rows, retrieval_pairs=retrieval_pairs, answers=answers)
