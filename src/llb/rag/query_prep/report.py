"""A/B report for the query-prep lane (validate-retrieval / compare-retrieval --query-prep-ab).

Scores retrieval at every cumulative stage (`baseline`, `+normalize`, `+typos`, ...) and
attributes a per-step marginal recall@k / MRR delta so nobody turns the lane on blind.
"""

from collections.abc import Callable, Iterable
from typing import Any

from llb.rag.query_prep.base import STEP_TYPOS, KnownWordProbe, Rewriter
from llb.rag.query_prep.glossary import Glossary
from llb.rag.query_prep.pipeline import QueryPrep

# (question, gold source spans) -- the per-item A/B input (matches `llb.rag.compare.CompareItem`).
AbItem = tuple[str, list[Any]]
# A retriever seam: processed query + k -> ranked chunk records (any RAG-store `.retrieve`).
RetrieveFn = Callable[[str, int], list[Any]]

AB_BASELINE_LABEL = "baseline"


def cumulative_pipelines(
    steps: Iterable[str],
    *,
    vocabulary: "frozenset[str] | None" = None,
    glossary: Glossary | None = None,
    rewriter: Rewriter | None = None,
    known_word: KnownWordProbe | None = None,
) -> list[tuple[str, "QueryPrep"]]:
    """`baseline` (no steps) then one pipeline per cumulative prefix (`+normalize`, `+typos`, ...).

    The A/B report scores each stage so a per-step marginal retrieval delta is attributable. Every
    prefix reuses the same resolved dependencies (`known_word` only binds to prefixes that
    include the typos step).
    """
    ordered = tuple(steps)
    stages: list[tuple[str, QueryPrep]] = [(AB_BASELINE_LABEL, QueryPrep.build(()))]
    for index in range(1, len(ordered) + 1):
        prefix = ordered[:index]
        pipeline = QueryPrep.build(
            prefix,
            vocabulary=vocabulary,
            glossary=glossary,
            rewriter=rewriter,
            known_word=known_word if STEP_TYPOS in prefix else None,
        )
        stages.append((f"+{ordered[index - 1]}", pipeline))
    return stages


def query_prep_ab_report(
    items: list[AbItem],
    retrieve: RetrieveFn,
    k: int,
    stages: list[tuple[str, "QueryPrep"]],
) -> dict[str, Any]:
    """Score retrieval at every cumulative stage and attribute per-step recall@k / MRR deltas.

    Pure over the injected `retrieve` seam (fake store in tests). Each stage's delta is measured
    against the PREVIOUS stage, so the marginal contribution of each added step is explicit.
    """
    from llb.rag.retrieval import evaluate_retrieval

    rows: list[dict[str, Any]] = []
    prev: dict[str, float] | None = None
    for label, pipeline in stages:
        pairs = [
            (retrieve(pipeline.process(question).processed, k), spans) for question, spans in items
        ]
        metrics = evaluate_retrieval(pairs, k)
        row: dict[str, Any] = {
            "stage": label,
            "recall_at_k": metrics["recall_at_k"],
            "mrr": metrics["mrr"],
            "delta_recall": 0.0 if prev is None else metrics["recall_at_k"] - prev["recall_at_k"],
            "delta_mrr": 0.0 if prev is None else metrics["mrr"] - prev["mrr"],
        }
        rows.append(row)
        prev = {"recall_at_k": metrics["recall_at_k"], "mrr": metrics["mrr"]}
    return {"k": k, "n": len(items), "stages": rows}


def format_query_prep_ab(report: dict[str, Any]) -> str:
    """Render the A/B stages as an ASCII table (AGENTS.md: ASCII-only, no box-drawing)."""
    lines = [f"[query-prep A/B] n={report['n']} k={report['k']}"]
    width = max((len(row["stage"]) for row in report["stages"]), default=len("stage"))
    lines.append(f"  {'stage'.ljust(width)}   recall@k   d(recall)      mrr    d(mrr)")
    for row in report["stages"]:
        lines.append(
            f"  {row['stage'].ljust(width)}   {row['recall_at_k']:8.3f}  {row['delta_recall']:+8.3f} "
            f"{row['mrr']:8.3f} {row['delta_mrr']:+8.3f}"
        )
    return "\n".join(lines)
