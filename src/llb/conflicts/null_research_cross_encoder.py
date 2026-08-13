"""Re-score the ranked rows and the frozen controls with a cross-encoder instead of one cosine.

Every geometry the third generation tried was a linear re-expression of ONE bi-encoder space, so
none of them could add information the encoder had already discarded. A cross-encoder reads both
passages together and is the smallest genuine capacity increase available on this host.

The lane is deliberately a RE-scorer: a cross-encoder cannot price a million-pair space, so it
scores the cosine shortlist and the frozen in-support controls. That bounds what it can claim -- a
threshold fitted on those controls inherits their unit count -- so the lane reports calibration
against the adjudicated labels and clustered tail coverage beside relation recall, and a fixture-F1
improvement alone can never accept it.
"""

from dataclasses import dataclass

from llb.conflicts.null_research_advanced import candidate_gates, clustered_tail_payload
from llb.conflicts.interval_stats import wilson_interval
from llb.conflicts.null_research_evaluation import (
    FIXTURE_POSITIVE_DOC_PAIRS,
    fixture_metrics,
    paired_transfer_payload,
    threshold_for_fpr,
)
from llb.conflicts.null_research_geometry import CorpusGeometry, DocPair
from llb.conflicts.null_research_precision import CandidateRow
from llb.conflicts.null_research_synthesis import SynthesizedControl
from llb.core.contracts.common import JsonObject
from llb.rag.rerank import RerankScorer

CROSS_ENCODER_METHOD = "cross_encoder_relation"
CALIBRATION_BINS = 4
MIN_CALIBRATION_ROWS = 8


@dataclass(frozen=True)
class ScoredRows:
    """One corpus's shortlist in both spaces, plus the labels the adjudicator gave those rows."""

    rows: list[CandidateRow]
    cosines: list[float]
    scores: list[float]
    actionable: list[bool | None]

    def document_maxima(self, corpus: CorpusGeometry) -> dict[DocPair, float]:
        maxima: dict[DocPair, float] = {}
        for row, score in zip(self.rows, self.scores):
            pair = row.document_pair(corpus)
            maxima[pair] = max(maxima.get(pair, score), score)
        return maxima

    def labelled(self) -> list[tuple[float, bool]]:
        return [
            (score, bool(label))
            for score, label in zip(self.scores, self.actionable)
            if label is not None
        ]


def score_pairs(scorer: RerankScorer, pairs: list[tuple[str, str]]) -> list[float]:
    """Cross-encoder score per (left, right) passage pair, batched by left passage."""
    grouped: dict[str, list[int]] = {}
    for position, (left, _) in enumerate(pairs):
        grouped.setdefault(left, []).append(position)
    scores = [0.0] * len(pairs)
    for left, positions in grouped.items():
        for position, value in zip(
            positions, scorer(left, [pairs[index][1] for index in positions])
        ):
            scores[position] = float(value)
    return scores


def score_shortlist(
    corpus: CorpusGeometry,
    rows: list[CandidateRow],
    verdicts: list[JsonObject],
    scorer: RerankScorer,
) -> ScoredRows:
    """Re-score the cosine shortlist and carry each row's adjudicated label alongside it."""
    pairs = [(corpus.chunks[row.left]["text"], corpus.chunks[row.right]["text"]) for row in rows]
    labels: list[bool | None] = [
        bool(verdict["actionable"]) if verdict.get("parsed") else None
        for verdict in verdicts[: len(rows)]
    ]
    labels.extend([None] * (len(rows) - len(labels)))
    return ScoredRows(
        rows=rows,
        cosines=[row.score for row in rows],
        scores=score_pairs(scorer, pairs),
        actionable=labels,
    )


def score_controls(
    controls: list[SynthesizedControl], scorer: RerankScorer
) -> tuple[list[float], list[float]]:
    """Control scores for the frozen generated bank, plus one maximum per source unit."""
    scores = score_pairs(scorer, [(control.source_text, control.text) for control in controls])
    by_source: dict[int, float] = {}
    for control, score in zip(controls, scores):
        by_source[control.source_ordinal] = max(by_source.get(control.source_ordinal, score), score)
    return sorted(scores), sorted(by_source.values())


def calibration_curve(scored: ScoredRows) -> JsonObject:
    """Does a higher cross-encoder score mean a higher adjudicated conflict rate?"""
    labelled = sorted(scored.labelled())
    if len(labelled) < MIN_CALIBRATION_ROWS:
        return {"labelled_rows": len(labelled), "bins": [], "monotone": False, "resolved": False}
    width = max(1, len(labelled) // CALIBRATION_BINS)
    bins: list[JsonObject] = []
    for start in range(0, len(labelled), width):
        block = labelled[start : start + width]
        if not block:
            continue
        actionable = sum(label for _, label in block)
        lower, upper = wilson_interval(actionable, len(block))
        bins.append(
            {
                "score_range": [round(block[0][0], 6), round(block[-1][0], 6)],
                "rows": len(block),
                "actionable_rows": actionable,
                "actionable_fraction": round(actionable / len(block), 6),
                "wilson_95": [round(lower, 6), round(upper, 6)],
            }
        )
    fractions = [float(payload["actionable_fraction"]) for payload in bins]
    return {
        "labelled_rows": len(labelled),
        "bins": bins,
        "monotone": all(earlier <= later for earlier, later in zip(fractions, fractions[1:])),
        "resolved": True,
    }


def _relation_recovery(scored: ScoredRows, threshold: float) -> JsonObject:
    """Of the rows the adjudicator called conflicts, how many the cross-encoder threshold keeps."""
    labelled = scored.labelled()
    positives = [score for score, label in labelled if label]
    kept = sum(score >= threshold for score in positives)
    negatives = [score for score, label in labelled if not label]
    return {
        "adjudicated_conflict_rows": len(positives),
        "conflict_rows_kept": kept,
        "relation_recall": round(kept / len(positives), 6) if positives else None,
        "non_conflict_rows_kept": sum(score >= threshold for score in negatives),
        "non_conflict_rows": len(negatives),
    }


def cross_encoder_candidate(
    corpora: dict[str, CorpusGeometry],
    scored: dict[str, ScoredRows],
    controls: dict[str, tuple[list[float], list[float]]],
    effective_units: dict[str, int],
    rank: JsonObject,
    feasibility: JsonObject,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    seed: int,
) -> JsonObject:
    """One acceptance row for the cross-encoder scorer, on the same gates every lane answers."""
    thresholds = {
        dataset: threshold_for_fpr(scores, fpr) for dataset, (scores, _) in controls.items()
    }
    tails = {
        dataset: clustered_tail_payload(
            controls[dataset][0],
            controls[dataset][1],
            thresholds[dataset],
            fpr,
            effective_units[dataset],
            seed + position,
        )
        for position, dataset in enumerate(controls)
    }
    curves = {dataset: calibration_curve(payload) for dataset, payload in scored.items()}
    diagnostics = {
        dataset: {
            **curves[dataset],
            **_relation_recovery(scored[dataset], thresholds[dataset]),
            "scored_rows": len(scored[dataset].rows),
            "calibrated": bool(curves[dataset]["resolved"] and curves[dataset]["monotone"]),
        }
        for dataset in scored
    }
    fixture = fixture_metrics(
        scored["fixture"].document_maxima(corpora["fixture"]),
        thresholds["fixture"],
        FIXTURE_POSITIVE_DOC_PAIRS,
    )
    transfers = {
        dataset: paired_transfer_payload(
            scored[dataset].cosines, scored[dataset].scores, thresholds[dataset], transfer_threshold
        )
        for dataset in ("hr", "goods")
    }
    gates = candidate_gates(
        fixture,
        rank,
        transfers["hr"],
        transfers["goods"],
        tails,
        diagnostics,
        max_goods_candidates=max_goods_candidates,
        eligible=True,
        control_key="calibrated",
        extra={"operating_point_feasible": bool(feasibility["feasible"])},
    )
    return {
        "method": CROSS_ENCODER_METHOD,
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "null_tails": tails,
        "diagnostics": diagnostics,
        "fixture": fixture,
        "hr": transfers["hr"],
        "goods": transfers["goods"],
        "gates": gates,
    }
