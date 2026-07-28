"""Context-ablation adapter for the shared paired-power contract."""

from pathlib import Path
from typing import Any, cast

from llb.eval.context_ablation.models import (
    DERIVED_LONG_CONTEXT_DELTA,
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    ContextAblationReport,
    LongContextPowerAnalysis,
)
from llb.rag.fusion_evidence.power import (
    DEFAULT_TARGET_POWER,
    plan_from_artifact as shared_plan_from_artifact,
    required_sample_size,
    resolve_power_analysis as shared_resolve_power_analysis,
    write_power_plan,
)


def _reference_deltas(payload: dict[str, Any]) -> list[float]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("power reference has no per-item ledger")
    skipped = set(payload.get("lanes", {}).get(LANE_LONG_CONTEXT, {}).get("skipped_item_ids", []))
    use_fitting = any(
        entry.get("label") == DERIVED_LONG_CONTEXT_DELTA_FITTING
        for entry in payload.get("derived", [])
    )
    deltas: list[float] = []
    for item in items:
        if use_fitting and item.get("item_id") in skipped:
            continue
        lanes = item.get("lanes", {})
        try:
            long_score = float(lanes[LANE_LONG_CONTEXT]["objective_score"])
            rag_score = float(lanes[LANE_RAG]["objective_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "power reference lacks paired long_context and rag objectives"
            ) from exc
        deltas.append(long_score - rag_score)
    return deltas


def plan_from_artifact(
    reference_artifact: Path,
    *,
    minimum_detectable_delta: float,
    target_power: float,
    confidence: float,
    planned_n: int,
) -> LongContextPowerAnalysis:
    """Build the current long-context declaration through the shared statistics seam."""
    plan = shared_plan_from_artifact(
        reference_artifact,
        _reference_deltas,
        minimum_detectable_delta=minimum_detectable_delta,
        target_power=target_power,
        confidence=confidence,
        planned_n=planned_n,
        selector={
            "lane": "compare-context-strategies",
            "candidate": LANE_LONG_CONTEXT,
            "baseline": LANE_RAG,
            "metric": "objective_score",
            "population": "eligible",
        },
    )
    return cast(LongContextPowerAnalysis, plan)


def resolve_power_analysis(
    report: ContextAblationReport, plan: LongContextPowerAnalysis
) -> LongContextPowerAnalysis:
    """Resolve the selected long-context delta with realized variance."""
    by_label = {entry["label"]: entry for entry in report["derived"]}
    entry = by_label.get(DERIVED_LONG_CONTEXT_DELTA_FITTING) or by_label.get(
        DERIVED_LONG_CONTEXT_DELTA
    )
    if entry is None:
        result = dict(plan)
        result.update(
            resolution="undecidable",
            direction="none",
            reason="no long-context delta was scored",
        )
        return cast(LongContextPowerAnalysis, result)
    skipped = set(report["lanes"][LANE_LONG_CONTEXT]["skipped_item_ids"])
    use_fitting = entry["label"] == DERIVED_LONG_CONTEXT_DELTA_FITTING
    deltas = [
        item["lanes"][LANE_LONG_CONTEXT]["objective_score"]
        - item["lanes"][LANE_RAG]["objective_score"]
        for item in report["items"]
        if not use_fitting or item["item_id"] not in skipped
    ]
    return cast(
        LongContextPowerAnalysis,
        shared_resolve_power_analysis(
            dict(plan),
            deltas,
            entry["paired"],
            candidate=LANE_LONG_CONTEXT,
            baseline=LANE_RAG,
        ),
    )


__all__ = [
    "DEFAULT_TARGET_POWER",
    "plan_from_artifact",
    "required_sample_size",
    "resolve_power_analysis",
    "write_power_plan",
]
