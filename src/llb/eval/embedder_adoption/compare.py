"""Pure per-cell comparison of two encoders; the keep-or-extend call itself lives in `verdict.py`.

File-driven: the input is one list of canonical `scores.jsonl` rows per (cell, encoder), so the
whole comparison is unit-tested with dict rows -- no backend, no store, no GPU. Item alignment
reuses `llb.eval.paired_cases` and the statistics reuse the fusion-evidence paired bootstrap, so a
cell's interval is read exactly like every other paired interval in the repo.

ONE set of resample indexes is drawn for the whole sweep and reused by every cell and metric
(common random numbers), which is what makes the cells comparable to each other rather than each
carrying its own sampling noise. That same draw also measures each cell's borderline annotation
(`llb.eval.embedder_adoption.stability`), so the sweep an operator runs on ONE model already says
whether its verdict rests on a knife-edge row -- it does not take assembling a roster to find out.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from llb.eval.embedder_adoption.models import (
    CELL_METRICS,
    COLUMN_FIRST_HIT_RANK,
    METRIC_OBJECTIVE,
    METRIC_RECIPROCAL_RANK,
    AdoptionBarReport,
    CellReport,
    CellSpec,
    ItemDeltas,
    LaneMetrics,
)
from llb.eval.embedder_adoption.stability import RowStability, stability_from_index_sets
from llb.eval.embedder_adoption.verdict import decide_bar
from llb.eval.paired_cases import CaseRows, lane_vectors, shared_item_ids
from llb.rag.fusion_evidence.slices import MetricVectors
from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    TIGHTER_CONFIDENCE,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_index_sets,
    bootstrap_interval,
    paired_comparison,
)

logger = logging.getLogger(__name__)

# One cell's scored rows, keyed by encoder model id.
CellRows = Mapping[str, CaseRows]

# The two metrics the three-state reading is built from, so a sweep restricted to a narrower metric
# list still reports its intervals -- it just carries no borderline annotation.
_STABILITY_METRICS = (METRIC_OBJECTIVE, METRIC_RECIPROCAL_RANK)


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
        "verdict": decide_bar(
            reports, baseline=baseline, candidate=candidate, confidence=confidence
        ),
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
    stability = _cell_stability(
        item_ids, vectors[candidate], vectors[baseline], index_sets, confidence
    )
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
        **({"stability": stability} if stability is not None else {}),  # type: ignore[typeddict-item]
    }


def _cell_stability(
    item_ids: Sequence[str],
    candidate: MetricVectors,
    baseline: MetricVectors,
    index_sets: list[list[int]],
    confidence: float,
) -> RowStability | None:
    """How settled this cell's reading is, from the vectors and the draw the intervals used.

    Returns `None` in the two cases where the annotation would not mean what it says: no resamples
    were drawn (`p_positive` is a share OF resamples, and a zero would read as a confident
    negative), or the reporting level sits outside the two conventional neighbours the flag is
    defined against. The deltas are in the shared sorted item order, which is the same order the
    roster's bundle re-derivation produces -- that is what makes the two measurements identical.
    """
    if not index_sets:
        return None
    if not LOOSER_CONFIDENCE < confidence < TIGHTER_CONFIDENCE:
        logger.warning(
            "[compare-embedder-adoption] no borderline annotation at confidence %.3f: the flag is "
            "defined against the neighbouring %.3f / %.3f conventions and this level is outside "
            "them",
            confidence,
            LOOSER_CONFIDENCE,
            TIGHTER_CONFIDENCE,
        )
        return None
    if not all(metric in candidate and metric in baseline for metric in _STABILITY_METRICS):
        return None
    deltas = ItemDeltas(
        item_ids=list(item_ids),
        objective=_delta(candidate, baseline, METRIC_OBJECTIVE),
        reciprocal_rank=_delta(candidate, baseline, METRIC_RECIPROCAL_RANK),
    )
    return stability_from_index_sets(deltas, index_sets, confidence=confidence)


def _delta(candidate: MetricVectors, baseline: MetricVectors, metric: str) -> list[float]:
    return [c - b for c, b in zip(candidate[metric], baseline[metric])]
