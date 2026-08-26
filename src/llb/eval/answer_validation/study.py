"""Compare the three validation lanes on one item set (pure, file-driven).

The obvious failure mode of any validator is refusing correct work, so this study is shaped to
make that visible rather than to make the gate look good:

  - CATCH and FALSE REJECTION are separate numbers, per axiom class, read off the gated lane's own
    rows: a refused answer the reference scores correct is a false rejection, whatever else the
    gate got right.
  - the objective delta is read on the COMMONLY ANSWERED items -- the items every lane ended `ok`
    on -- because a gate that improves the mean by declining the hard items would otherwise look
    like a win. Abstention rate and answered count are reported beside it so the decline is
    readable as the decline it is.
  - the cost is reported per ANSWER, in tokens and seconds, so the repair round trip is priced
    rather than assumed cheap.

Everything is computed from canonical per-case rows, so the whole study runs in CI over dict rows:
no backend, no store, no GPU.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from typing_extensions import TypedDict

from llb.eval import common as eval_common
from llb.eval.answer_validation.constants import (
    LANE_OFF,
    REFERENCE_CORRECT_COLUMN,
)
from llb.eval.answer_validation.verdict import class_verdicts, lane_decision
from llb.eval.paired_cases import CaseRows, rows_by_item, shared_item_ids
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_index_sets,
)

# The correctness columns reported per lane. `objective_score` is the headline; `contains` is the
# found-rate the catch / false-rejection split is decided on, because the token-F1 objective mixes
# needle-finding with terseness and would price a verbose correct answer as a wrong one.
CORRECTNESS_COLUMNS = ("objective_score", REFERENCE_CORRECT_COLUMN, "ranking_score")
COST_COLUMNS = ("completion_tokens", "prompt_tokens", "latency_s")


class LaneSummary(TypedDict):
    """One lane read on its own terms: what it answered, what it refused, what it cost."""

    n: int
    n_answered: int
    answered_rate: float
    abstention_rate: float
    ontology_violation_rate: float
    envelope_repair_rate: float
    validation_repair_rate: float
    objective_score: float
    contains: float
    ranking_score: float
    completion_tokens: float
    prompt_tokens: float
    latency_s: float
    run_dirs: list[str]


class Refusal(TypedDict):
    """One answer the gate refused, with everything needed to adjudicate the refusal by hand.

    The catch / false-rejection split rests on an automated correctness proxy, and a proxy can be
    wrong -- so every rejection is listed rather than only counted. A rejection nobody can inspect
    is not evidence.
    """

    item_id: str
    axiom_classes: list[str]
    axiom_ids: list[str]
    labelled: str  # CATCH | FALSE_REJECTION, as the reference proxy scored it
    contains: float
    objective_score: float
    repaired: bool
    completion_tokens: int
    answer_preview: str


LABEL_CATCH = "catch"
LABEL_FALSE_REJECTION = "false_rejection"


class LaneReading(TypedDict):
    """One candidate lane against the baseline on the items both of them answered."""

    lane: str
    baseline: str
    n_commonly_answered: int
    objective_delta: PairedComparison
    contains_delta: PairedComparison
    added_completion_tokens: float
    added_latency_s: float
    decision: str
    reason: str


def lane_summary(rows: CaseRows, run_dirs: Sequence[str] = ()) -> LaneSummary:
    """Everything one lane says about itself, over every case it scored."""
    n = len(rows)
    answered = [row for row in rows if str(row.get("status")) == eval_common.OK]
    return {
        "n": n,
        "n_answered": len(answered),
        "answered_rate": _share(rows, lambda row: str(row.get("status")) == eval_common.OK),
        "abstention_rate": _share(rows, _abstained),
        "ontology_violation_rate": _share(
            rows, lambda row: str(row.get("status")) == eval_common.ONTOLOGY_VIOLATION
        ),
        "envelope_repair_rate": _share(rows, lambda row: bool(row.get("repaired"))),
        "validation_repair_rate": _share(rows, lambda row: bool(row.get("validation_repaired"))),
        **{column: _mean(rows, column) for column in CORRECTNESS_COLUMNS},  # type: ignore[typeddict-item]
        **{column: _mean(rows, column) for column in COST_COLUMNS},
        "run_dirs": list(run_dirs),
    }


def commonly_answered(
    lanes: Mapping[str, CaseRows], item_ids: Sequence[str] | None = None
) -> list[str]:
    """The items EVERY lane ended `ok` on -- the only set an objective delta may be read on.

    A gate that refuses the hard items would otherwise raise the mean of the lane it is compared
    against, which is the exact reading this study exists to prevent.
    """
    item_ids = list(item_ids) if item_ids is not None else shared_item_ids(lanes)
    by_lane = {label: rows_by_item(rows) for label, rows in lanes.items()}
    return [
        item_id
        for item_id in item_ids
        if all(
            str(rows.get(item_id, {}).get("status")) == eval_common.OK for rows in by_lane.values()
        )
    ]


def lane_reading(
    lane: str,
    lanes: Mapping[str, CaseRows],
    item_ids: Sequence[str],
    baseline: str,
    index_sets: list[list[int]],
    confidence: float,
) -> LaneReading:
    """One candidate lane's paired reading against the baseline on the commonly-answered items."""
    from llb.rag.fusion_evidence.paired import regresses, separates

    candidate_rows = rows_by_item(lanes[lane])
    baseline_rows = rows_by_item(lanes[baseline])
    objective = _paired(
        candidate_rows, baseline_rows, item_ids, "objective_score", index_sets, confidence
    )
    contains = _paired(
        candidate_rows, baseline_rows, item_ids, REFERENCE_CORRECT_COLUMN, index_sets, confidence
    )
    added_tokens = _added(candidate_rows, baseline_rows, item_ids, "completion_tokens")
    added_latency = _added(candidate_rows, baseline_rows, item_ids, "latency_s")
    decision, reason = lane_decision(
        lane, objective, separates(objective, confidence), regresses(objective, confidence)
    )
    return {
        "lane": lane,
        "baseline": baseline,
        "n_commonly_answered": len(item_ids),
        "objective_delta": objective,
        "contains_delta": contains,
        "added_completion_tokens": round(added_tokens, 2),
        "added_latency_s": round(added_latency, 3),
        "decision": decision,
        "reason": reason,
    }


def refusals(rows: CaseRows) -> list[Refusal]:
    """Every case the gate refused, in item order, with the label the reference proxy gave it."""
    refused = [row for row in rows if str(row.get("status")) == eval_common.ONTOLOGY_VIOLATION]
    return [
        {
            "item_id": str(row["item_id"]),
            "axiom_classes": [str(name) for name in row.get("validation_classes", [])],
            "axiom_ids": [str(name) for name in row.get("validation_axioms", [])],
            "labelled": (
                LABEL_FALSE_REJECTION
                if float(row.get(REFERENCE_CORRECT_COLUMN, 0.0) or 0.0) >= 1.0
                else LABEL_CATCH
            ),
            "contains": float(row.get(REFERENCE_CORRECT_COLUMN, 0.0) or 0.0),
            "objective_score": float(row.get("objective_score", 0.0) or 0.0),
            "repaired": bool(row.get("validation_repaired")),
            "completion_tokens": int(row.get("completion_tokens", 0) or 0),
            "answer_preview": str(row.get("answer_preview", "")),
        }
        for row in sorted(refused, key=lambda row: str(row["item_id"]))
    ]


def analyze(
    lanes: Mapping[str, CaseRows],
    *,
    baseline: str = LANE_OFF,
    run_dirs: Mapping[str, list[str]] | None = None,
    gated_lane: str | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """The whole comparison: per-lane summaries, paired readings, and per-class verdicts."""
    if baseline not in lanes:
        raise ValueError(f"the baseline lane {baseline!r} was not scored: {sorted(lanes)}")
    if len(lanes) < 2:
        raise ValueError("the comparison needs the baseline lane and at least one candidate lane")
    dirs = dict(run_dirs or {})
    scored_ids = shared_item_ids(lanes)
    item_ids = commonly_answered(lanes, scored_ids)
    index_sets = bootstrap_index_sets(len(item_ids), resamples, seed) if item_ids else []
    gated_rows = list(lanes.get(gated_lane or "", []))
    gate_index_sets = bootstrap_index_sets(len(gated_rows), resamples, seed) if gated_rows else []
    return {
        "baseline": baseline,
        "gated_lane": gated_lane,
        "n_items": len(scored_ids),
        "n_commonly_answered": len(item_ids),
        "commonly_answered": item_ids,
        "lanes": {label: lane_summary(rows, dirs.get(label, [])) for label, rows in lanes.items()},
        "readings": [
            lane_reading(label, lanes, item_ids, baseline, index_sets, confidence)
            for label in lanes
            if label != baseline
        ],
        "axiom_classes": class_verdicts(gated_rows, gate_index_sets, confidence),
        "refusals": refusals(gated_rows),
        "settings": {"resamples": resamples, "confidence": confidence, "seed": seed},
    }


def _paired(
    candidate: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    item_ids: Sequence[str],
    column: str,
    index_sets: list[list[int]],
    confidence: float,
) -> PairedComparison:
    return paired_comparison(
        [float(candidate[item_id].get(column, 0.0) or 0.0) for item_id in item_ids],
        [float(baseline[item_id].get(column, 0.0) or 0.0) for item_id in item_ids],
        index_sets,
        confidence,
    )


def _added(
    candidate: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    item_ids: Sequence[str],
    column: str,
) -> float:
    if not item_ids:
        return 0.0
    return sum(
        float(candidate[item_id].get(column, 0.0) or 0.0)
        - float(baseline[item_id].get(column, 0.0) or 0.0)
        for item_id in item_ids
    ) / len(item_ids)


def _abstained(row: Mapping[str, Any]) -> bool:
    """Whether this case declined to answer, declared or classified.

    The `off` lane has no declaration to read, so its abstention is its `refusal` status -- the
    same signal the free-text path has always used. Reading only the declared flag would report
    every free-text refusal as an answer and make the gated lane look uniquely reticent.
    """
    return bool(row.get("envelope_abstained")) or str(row.get("status")) == eval_common.REFUSAL


def _share(rows: CaseRows, predicate: Callable[[Mapping[str, Any]], bool]) -> float:
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4) if rows else 0.0


def _mean(rows: CaseRows, column: str) -> float:
    values = [float(row.get(column, 0.0) or 0.0) for row in rows]
    return round(sum(values) / len(values), 4) if values else 0.0
