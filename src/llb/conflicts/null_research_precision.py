"""Claim-tier precision: the quantity an operator can actually act on.

A semantic false-positive rate needs a null the corpus cannot supply. The precision of the list the
operator adjudicates does not: it is estimated on the rows themselves, so its sample size is the
adjudication budget rather than the pair space. This lane ranks the candidate rows exactly as the
semantic tier would, adjudicates them with the local Ukrainian model, calibrates that adjudicator
against the planted fixture's frozen relations, and reports precision with a TWO-WAY clustered lower
confidence bound -- rows that share a left or right chunk are not independent evidence.
"""

from dataclasses import dataclass

import numpy as np

from llb.conflicts.claim_prompt import AdjudicationError, adjudication_prompt, parse_adjudication
from llb.conflicts.constants import REL_COMPLEMENTARY
from llb.conflicts.null_research_clusters import two_way_proportion_bound
from llb.conflicts.null_research_evaluation import (
    FIXTURE_POSITIVE_DOC_PAIRS,
    wilson_interval,
)
from llb.conflicts.null_research_geometry import CorpusGeometry
from llb.core.contracts.common import JsonObject
from llb.prep.frontier_telemetry import LLMComplete

PRECISION_BUDGETS = (6, 12, 25, 50)
MIN_ADJUDICATOR_ACCURACY_LCB = 0.60
MIN_PRECISION_LCB = 0.50
MAX_UNPARSED_FRACTION = 0.05


@dataclass(frozen=True)
class CandidateRow:
    """One ranked cross-document chunk pair, addressed the way the semantic tier addresses it."""

    left: int
    right: int
    score: float

    def key(self, corpus: CorpusGeometry) -> tuple[str, str]:
        return (
            str(corpus.chunks[self.left].get("chunk_id", self.left)),
            str(corpus.chunks[self.right].get("chunk_id", self.right)),
        )

    def document_pair(self, corpus: CorpusGeometry) -> tuple[str, str]:
        pair = sorted((corpus.chunks[self.left]["doc_id"], corpus.chunks[self.right]["doc_id"]))
        return pair[0], pair[1]


def top_candidate_rows(corpus: CorpusGeometry, budget: int) -> list[CandidateRow]:
    """The highest-cosine cross-document chunk pairs, in the semantic tier's own ranking."""
    matrix = np.asarray([corpus.vectors.row(index) for index in corpus.allowed], dtype="float64")
    doc_ids = [corpus.chunks[index]["doc_id"] for index in corpus.allowed]
    codes = {doc_id: code for code, doc_id in enumerate(sorted(set(doc_ids)))}
    labels = np.asarray([codes[doc_id] for doc_id in doc_ids])
    similarity = matrix @ matrix.T
    comparable = np.triu(np.ones_like(similarity, dtype=bool), k=1) & (
        labels[:, None] != labels[None, :]
    )
    take = min(budget, int(comparable.sum()))
    if take < 1:
        return []
    scores = np.where(comparable, similarity, -np.inf).reshape(-1)
    selected = np.argpartition(scores, -take)[-take:]
    ordered = selected[np.argsort(scores[selected], kind="stable")][::-1]
    return [
        CandidateRow(
            left=corpus.allowed[int(index) // len(matrix)],
            right=corpus.allowed[int(index) % len(matrix)],
            score=float(scores[index]),
        )
        for index in ordered.tolist()
    ]


def adjudicate_rows(
    corpus: CorpusGeometry, rows: list[CandidateRow], complete: LLMComplete
) -> list[JsonObject]:
    """One local-model relation verdict per ranked row, retaining unparsed rows as failures."""
    verdicts: list[JsonObject] = []
    for rank, row in enumerate(rows, start=1):
        prompt = adjudication_prompt(
            corpus.chunks[row.left]["text"], corpus.chunks[row.right]["text"]
        )
        try:
            parsed = parse_adjudication(complete(prompt))
        except (AdjudicationError, RuntimeError) as exc:
            verdicts.append(
                {
                    "dataset": corpus.name,
                    "rank": rank,
                    "cosine": round(row.score, 6),
                    "relation": None,
                    "actionable": False,
                    "parsed": False,
                    "error": str(exc)[:200],
                }
            )
            continue
        verdicts.append(
            {
                "dataset": corpus.name,
                "rank": rank,
                "cosine": round(row.score, 6),
                "document_pair": list(row.document_pair(corpus)),
                "relation": parsed["relation"],
                "confidence": parsed["confidence"],
                "actionable": parsed["relation"] != REL_COMPLEMENTARY,
                "parsed": True,
            }
        )
    return verdicts


def precision_curve(
    corpus: CorpusGeometry,
    rows: list[CandidateRow],
    verdicts: list[JsonObject],
    *,
    seed: int,
) -> list[JsonObject]:
    """Precision at each candidate budget with its two-way clustered lower bound."""
    keys = [row.key(corpus) for row in rows]
    curve: list[JsonObject] = []
    for budget in PRECISION_BUDGETS:
        if budget > len(rows):
            continue
        flags = [bool(verdict["actionable"]) for verdict in verdicts[:budget]]
        successes = sum(flags)
        lower, upper = wilson_interval(successes, budget)
        curve.append(
            {
                "budget": budget,
                "actionable_rows": successes,
                "precision": round(successes / budget, 6),
                "wilson_95": [round(lower, 6), round(upper, 6)],
                "two_way_clustered_lcb": round(
                    two_way_proportion_bound(
                        flags,
                        [key[0] for key in keys[:budget]],
                        [key[1] for key in keys[:budget]],
                        seed=seed + budget,
                    ),
                    6,
                ),
                "left_clusters": len({key[0] for key in keys[:budget]}),
                "right_clusters": len({key[1] for key in keys[:budget]}),
            }
        )
    return curve


def adjudicator_calibration(
    corpus: CorpusGeometry, rows: list[CandidateRow], verdicts: list[JsonObject]
) -> JsonObject:
    """Agreement between the local adjudicator and the fixture's frozen planted relations."""
    labelled = [
        (bool(verdict["actionable"]), row.document_pair(corpus) in FIXTURE_POSITIVE_DOC_PAIRS)
        for row, verdict in zip(rows, verdicts)
        if verdict["parsed"]
    ]
    agreements = sum(predicted == truth for predicted, truth in labelled)
    lower, upper = wilson_interval(agreements, len(labelled))
    positives = [predicted for predicted, truth in labelled if truth]
    negatives = [predicted for predicted, truth in labelled if not truth]
    return {
        "labelled_rows": len(labelled),
        "agreements": agreements,
        "accuracy": round(agreements / len(labelled), 6) if labelled else 0.0,
        "accuracy_wilson_95": [round(lower, 6), round(upper, 6)],
        "planted_positive_rows": len(positives),
        "planted_negative_rows": len(negatives),
        "recall_on_planted": round(sum(positives) / len(positives), 6) if positives else None,
        "specificity_on_planted": round(sum(not value for value in negatives) / len(negatives), 6)
        if negatives
        else None,
        "calibrated": bool(labelled) and lower >= MIN_ADJUDICATOR_ACCURACY_LCB,
    }


def _dataset_precision(
    corpus: CorpusGeometry, complete: LLMComplete, *, budget: int, seed: int
) -> tuple[JsonObject, list[CandidateRow], list[JsonObject]]:
    rows = top_candidate_rows(corpus, budget)
    verdicts = adjudicate_rows(corpus, rows, complete)
    unparsed = sum(not verdict["parsed"] for verdict in verdicts)
    return (
        {
            "adjudicated_rows": len(rows),
            "unparsed_rows": unparsed,
            "unparsed_fraction": round(unparsed / len(rows), 6) if rows else 1.0,
            "cosine_range": [round(rows[-1].score, 6), round(rows[0].score, 6)] if rows else None,
            "relations": {
                relation: sum(verdict["relation"] == relation for verdict in verdicts)
                for relation in sorted(
                    {str(verdict["relation"]) for verdict in verdicts if verdict["parsed"]}
                )
            },
            "precision_curve": precision_curve(corpus, rows, verdicts, seed=seed),
        },
        rows,
        verdicts,
    )


def claim_precision_lane(
    corpora: dict[str, CorpusGeometry],
    complete: LLMComplete,
    *,
    adjudication_budget: int,
    operating_budget: int,
    seed: int,
) -> JsonObject:
    """Rank, adjudicate, calibrate, and bound the precision of the list operators receive."""
    datasets: dict[str, JsonObject] = {}
    verdict_rows: list[JsonObject] = []
    calibration: JsonObject = {}
    for position, (name, corpus) in enumerate(corpora.items()):
        payload, rows, verdicts = _dataset_precision(
            corpus, complete, budget=adjudication_budget, seed=seed + position
        )
        datasets[name] = payload
        verdict_rows.extend(verdicts)
        if name == "fixture":
            calibration = adjudicator_calibration(corpus, rows, verdicts)
    operating = {
        name: next(
            (
                point
                for point in payload["precision_curve"]
                if int(point["budget"]) == operating_budget
            ),
            None,
        )
        for name, payload in datasets.items()
    }
    transfers = {
        name: float(point["two_way_clustered_lcb"]) >= MIN_PRECISION_LCB
        for name, point in operating.items()
        if point is not None and name != "fixture"
    }
    gates = {
        "adjudicator_calibrated": bool(calibration.get("calibrated")),
        "rows_adjudicated": all(
            float(payload["unparsed_fraction"]) <= MAX_UNPARSED_FRACTION
            for payload in datasets.values()
        ),
        "precision_bound_transfers": bool(transfers) and all(transfers.values()),
    }
    return {
        "method": "claim_tier_precision",
        "operating_budget": operating_budget,
        "adjudication_budget": adjudication_budget,
        "min_precision_lcb": MIN_PRECISION_LCB,
        "min_adjudicator_accuracy_lcb": MIN_ADJUDICATOR_ACCURACY_LCB,
        "adjudicator_calibration": calibration,
        "datasets": datasets,
        "operating_point": operating,
        "transfers": transfers,
        "verdicts": verdict_rows,
        "gates": {**gates, "accepted": all(gates.values())},
    }
