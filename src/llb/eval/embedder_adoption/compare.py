"""Pure per-cell comparison of two encoders, and the keep-or-extend call on the adoption bar.

File-driven: the input is one list of canonical `scores.jsonl` rows per (cell, encoder), so the
whole comparison is unit-tested with dict rows -- no backend, no store, no GPU. Item alignment
reuses `llb.eval.paired_cases` and the statistics reuse the fusion-evidence paired bootstrap, so a
cell's interval is read exactly like every other paired interval in the repo.

ONE set of resample indexes is drawn for the whole sweep and reused by every cell and metric
(common random numbers), which is what makes the cells comparable to each other rather than each
carrying its own sampling noise.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.eval.embedder_adoption.models import (
    CELL_METRICS,
    COLUMN_FIRST_HIT_RANK,
    DECISION_EXTEND_BAR,
    DECISION_KEEP_BAR,
    DECISION_NO_EVIDENCE,
    METRIC_OBJECTIVE,
    METRIC_RECIPROCAL_RANK,
    AdoptionBarReport,
    BarVerdict,
    CellReport,
    CellSpec,
    LaneMetrics,
)
from llb.eval.paired_cases import CaseRows, lane_vectors, shared_item_ids
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_index_sets,
    bootstrap_interval,
    format_interval,
    paired_comparison,
)

# One cell's scored rows, keyed by encoder model id.
CellRows = Mapping[str, CaseRows]


def with_reciprocal_rank(rows: CaseRows) -> CaseRows:
    """Add the `reciprocal_rank` column derived from each row's `first_hit_rank`.

    A missing rank means the retrieved context carried no gold span at all, which is reciprocal
    rank 0.0 -- the same convention `llb.rag.retrieval.reciprocal_rank` uses, so the answer-side
    column and the retrieval-side MRR the bake-off ranks on are the SAME statistic.
    """
    enriched: list[Mapping[str, Any]] = []
    for row in rows:
        rank = row.get(COLUMN_FIRST_HIT_RANK)
        value = 1.0 / int(rank) if rank else 0.0
        enriched.append({**row, METRIC_RECIPROCAL_RANK: value})
    return enriched


def compare_cells(
    cells: Sequence[tuple[CellSpec, CellRows]],
    run_dirs: Mapping[str, Mapping[str, list[str]]],
    *,
    baseline: str,
    candidate: str,
    metrics: Sequence[str] = CELL_METRICS,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> AdoptionBarReport:
    """Pair the candidate encoder against the baseline inside every cell, then decide the bar."""
    if not cells:
        raise ValueError("the sweep needs at least one cell")
    lanes = {
        f"{cell.label}/{model}": with_reciprocal_rank(rows)
        for cell, cell_rows in cells
        for model, rows in cell_rows.items()
    }
    item_ids = shared_item_ids(lanes)
    index_sets = bootstrap_index_sets(len(item_ids), resamples, seed)
    reports = [
        _cell_report(
            cell,
            {model: lanes[f"{cell.label}/{model}"] for model in cell_rows},
            run_dirs.get(cell.label, {}),
            item_ids=item_ids,
            baseline=baseline,
            candidate=candidate,
            metrics=metrics,
            index_sets=index_sets,
            confidence=confidence,
        )
        for cell, cell_rows in cells
    ]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "item_ids": item_ids,
        "metrics": list(metrics),
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "cells": reports,
        "verdict": decide_bar(reports, baseline=baseline, candidate=candidate),
    }


def _cell_report(
    cell: CellSpec,
    cell_rows: CellRows,
    run_dirs: Mapping[str, list[str]],
    *,
    item_ids: Sequence[str],
    baseline: str,
    candidate: str,
    metrics: Sequence[str],
    index_sets: list[list[int]],
    confidence: float,
) -> CellReport:
    missing = [model for model in (baseline, candidate) if model not in cell_rows]
    if missing:
        raise ValueError(f"cell {cell.label!r} did not score {', '.join(missing)}")
    vectors = {model: lane_vectors(rows, item_ids, metrics) for model, rows in cell_rows.items()}
    lanes: dict[str, LaneMetrics] = {
        model: {
            "run_dirs": list(run_dirs.get(model, [])),
            "metrics": {
                metric: bootstrap_interval(values[metric], index_sets, confidence)
                for metric in metrics
            },
        }
        for model, values in vectors.items()
    }
    return {
        "label": cell.label,
        "top_k": cell.top_k,
        "reranker": cell.reranker,
        "n": len(item_ids),
        "lanes": lanes,
        "paired": {
            metric: paired_comparison(
                vectors[candidate][metric], vectors[baseline][metric], index_sets, confidence
            )
            for metric in metrics
        },
    }


def _separated(cells: Sequence[CellReport], metric: str) -> list[str]:
    """Cell labels whose paired delta on `metric` lies wholly above zero."""
    return [cell["label"] for cell in cells if cell["paired"][metric]["delta"]["lo"] > 0.0]


def decide_bar(cells: Sequence[CellReport], *, baseline: str, candidate: str) -> BarVerdict:
    """Keep recall@k as the sole adoption bar, or extend it with the scoped first-hit-rank bar.

    Read in order, because the two negative outcomes are NOT the same finding:

    1. An objective delta clear of zero in any cell means the ranking advantage reached the answers
       in a real configuration -- that is the scope a second bar would carry, and it is the only
       result that justifies adding one.
    2. No such cell, but a reciprocal-rank delta clear of zero somewhere, is a MEASURED negative:
       the encoder does rank better inside the scored configuration and the answers do not move, so
       recall@k stays the sole bar.
    3. No cell separates on rank either. Then the premise the whole question rests on was never met
       in these configurations and the sweep decides nothing -- reporting that as "keep" would
       dress an unmet premise up as evidence.
    """
    answer_cells = _separated(cells, METRIC_OBJECTIVE)
    rank_cells = _separated(cells, METRIC_RECIPROCAL_RANK)
    verdict: BarVerdict = {
        "decision": DECISION_NO_EVIDENCE,
        "baseline": baseline,
        "candidate": candidate,
        "answer_cells": answer_cells,
        "rank_cells": rank_cells,
        "reason": "",
    }
    if answer_cells:
        detail = "; ".join(
            f"`{cell['label']}` objective "
            f"{format_interval(cell['paired'][METRIC_OBJECTIVE]['delta'])}"
            for cell in cells
            if cell["label"] in answer_cells
        )
        verdict["decision"] = DECISION_EXTEND_BAR
        verdict["reason"] = (
            f"`{candidate}` answers better than `{baseline}` in {len(answer_cells)} of "
            f"{len(cells)} cells ({detail}); a first-hit-rank gain is worth adopting under those "
            "configurations, so the bake-off gains a second bar scoped to them"
        )
        return verdict
    if rank_cells:
        best = max(cells, key=lambda cell: cell["paired"][METRIC_OBJECTIVE]["delta"]["mean"])
        verdict["decision"] = DECISION_KEEP_BAR
        verdict["reason"] = (
            f"`{candidate}` ranks the evidence earlier than `{baseline}` in "
            f"{', '.join(f'`{label}`' for label in rank_cells)} but no cell's objective interval "
            f"clears zero (best `{best['label']}` "
            f"{format_interval(best['paired'][METRIC_OBJECTIVE]['delta'])}); recall@k stays the "
            "sole adoption bar"
        )
        return verdict
    verdict["reason"] = (
        f"`{candidate}` does not separate from `{baseline}` on first-hit rank in ANY scored cell, "
        "so this sweep never tested the question the second bar would answer"
    )
    return verdict
