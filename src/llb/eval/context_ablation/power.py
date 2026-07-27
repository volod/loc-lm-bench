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
from llb.rag.fusion_evidence.evidence_gate import (
    minimum_discordant_pairs,
    reaches_reporting_level,
    resolving_item_count,
)
from llb.rag.fusion_evidence.paired import (
    compared_pairs,
    discordant_pairs,
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
    """Build the legacy-shaped declaration through the shared statistics seam."""
    plan = shared_plan_from_artifact(
        reference_artifact,
        _reference_deltas,
        minimum_detectable_delta=minimum_detectable_delta,
        target_power=target_power,
        confidence=confidence,
        planned_n=planned_n,
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
    # Older focused tests and third-party readers may pass only the derived row and the three
    # declaration fields the original resolver needed. Keep that pure resolution seam while full
    # artifacts use the shared realized-variance analysis below.
    required_plan_fields = {
        "reference_sample_sd",
        "target_power",
        "required_n",
        "planned_n",
    }
    if "items" not in report or not required_plan_fields.issubset(plan):
        return _legacy_resolution(entry["paired"], plan)
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


def _legacy_resolution(paired: Any, plan: LongContextPowerAnalysis) -> LongContextPowerAnalysis:
    """Original interval-only resolver for compact artifacts without a realized item ledger."""
    result = dict(plan)
    interval = paired["delta"]
    margin = plan["minimum_detectable_delta"]
    confidence = 1.0 - plan["alpha"]
    discordant = discordant_pairs(paired)
    resolvable = reaches_reporting_level(discordant, confidence)
    if interval["lo"] > 0.0 and resolvable:
        result.update(
            resolution="separated",
            direction=LANE_LONG_CONTEXT,
            reason="the paired interval is wholly above zero",
        )
    elif interval["hi"] < 0.0 and resolvable:
        result.update(
            resolution="separated",
            direction=LANE_RAG,
            reason="the paired interval is wholly below zero",
        )
    elif not resolvable and not (interval["lo"] >= -margin and interval["hi"] <= margin):
        required = resolving_item_count(discordant, compared_pairs(paired), confidence)
        clause = f"; at that discordance rate the level needs about {required} paired items"
        result.update(
            resolution="undecidable",
            direction="none",
            reason=(
                f"the two lanes differ on only {discordant} items, fewer than the "
                f"{minimum_discordant_pairs(confidence)} an exact sign test needs to reach this "
                f"level, so the interval cannot resolve a direction{clause}"
            ),
        )
    elif interval["lo"] >= -margin and interval["hi"] <= margin:
        result.update(
            resolution="flat",
            direction="neither",
            reason="the paired interval lies wholly inside the predeclared detectable-effect band",
        )
    else:
        size_note = (
            "the planned item count reached the power target"
            if plan["target_reached"]
            else "the planned item count did not reach the power target"
        )
        result.update(
            resolution="undecidable",
            direction="none",
            reason=f"the paired interval crosses zero and the detectable-effect band; {size_note}",
        )
    return cast(LongContextPowerAnalysis, result)


__all__ = [
    "DEFAULT_TARGET_POWER",
    "plan_from_artifact",
    "required_sample_size",
    "resolve_power_analysis",
    "write_power_plan",
]
