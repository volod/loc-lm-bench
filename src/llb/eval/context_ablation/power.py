"""A priori paired-power target and post-run resolution for the long-context delta.

The reference artifact supplies the variance, not the effect target: the operator predeclares the
smallest delta worth resolving, then the normal approximation prices the paired item count before
any new lane is scored. The resulting plan is persisted before inference starts.
"""

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

from llb.eval.context_ablation.models import (
    DERIVED_LONG_CONTEXT_DELTA,
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    POWER_RESOLUTION_FLAT,
    POWER_RESOLUTION_SEPARATED,
    POWER_RESOLUTION_UNDECIDABLE,
    ContextAblationReport,
    DerivedComparison,
    LongContextPowerAnalysis,
)
from llb.rag.fusion_evidence.evidence_gate import (
    minimum_discordant_pairs,
    reaches_reporting_level,
)
from llb.rag.fusion_evidence.stats import discordant_pairs

POWER_METHOD = "paired-normal-approximation"
DEFAULT_TARGET_POWER = 0.80


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("the power reference needs at least two paired items")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def required_sample_size(
    sample_sd: float,
    minimum_detectable_delta: float,
    *,
    alpha: float,
    target_power: float,
) -> int:
    """Paired two-sided normal-approximation item count, rounded up."""
    if sample_sd < 0.0:
        raise ValueError("reference sample SD must be non-negative")
    if minimum_detectable_delta <= 0.0:
        raise ValueError("minimum detectable delta must be positive")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    if not 0.5 < target_power < 1.0:
        raise ValueError("target power must be between 0.5 and 1")
    if sample_sd == 0.0:
        return 2
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(target_power)
    return max(2, math.ceil((critical * sample_sd / minimum_detectable_delta) ** 2))


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
    """Build the declaration from an earlier comparison and the planned new item count."""
    try:
        payload = json.loads(reference_artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read power reference {reference_artifact}: {exc}") from exc
    deltas = _reference_deltas(payload)
    sample_sd = _sample_sd(deltas)
    alpha = round(1.0 - confidence, 12)
    required_n = required_sample_size(
        sample_sd,
        minimum_detectable_delta,
        alpha=alpha,
        target_power=target_power,
    )
    return {
        "method": POWER_METHOD,
        "reference_artifact": str(reference_artifact),
        "reference_n": len(deltas),
        "reference_mean": sum(deltas) / len(deltas),
        "reference_sample_sd": sample_sd,
        "minimum_detectable_delta": minimum_detectable_delta,
        "target_power": target_power,
        "alpha": alpha,
        "required_n": required_n,
        "planned_n": planned_n,
        "target_reached": planned_n >= required_n,
    }


def _long_context_entry(report: ContextAblationReport) -> DerivedComparison | None:
    by_label = {entry["label"]: entry for entry in report["derived"]}
    return by_label.get(DERIVED_LONG_CONTEXT_DELTA_FITTING) or by_label.get(
        DERIVED_LONG_CONTEXT_DELTA
    )


def resolve_power_analysis(
    report: ContextAblationReport, plan: LongContextPowerAnalysis
) -> LongContextPowerAnalysis:
    """Attach a direction-aware separated, practically-flat, or undecidable reading."""
    result = dict(plan)
    entry = _long_context_entry(report)
    if entry is None:
        result.update(
            resolution=POWER_RESOLUTION_UNDECIDABLE,
            direction="none",
            reason="no long-context delta was scored",
        )
        return cast(LongContextPowerAnalysis, result)
    delta = entry["paired"]["delta"]
    margin = plan["minimum_detectable_delta"]
    confidence = 1.0 - plan["alpha"]
    # A one-sided bound clearing zero is a separation CLAIM, so it goes through the same gate every
    # other lane reads: an interval drawn from too few differing items cannot resolve a direction,
    # whichever side of zero it sits on.
    resolvable = reaches_reporting_level(discordant_pairs(entry["paired"]), confidence)
    if delta["lo"] > 0.0 and resolvable:
        result.update(
            resolution=POWER_RESOLUTION_SEPARATED,
            direction=LANE_LONG_CONTEXT,
            reason="the paired interval is wholly above zero",
        )
    elif delta["hi"] < 0.0 and resolvable:
        result.update(
            resolution=POWER_RESOLUTION_SEPARATED,
            direction=LANE_RAG,
            reason="the paired interval is wholly below zero",
        )
    elif not resolvable and not (delta["lo"] >= -margin and delta["hi"] <= margin):
        result.update(
            resolution=POWER_RESOLUTION_UNDECIDABLE,
            direction="none",
            reason=(
                f"the two lanes differ on only {discordant_pairs(entry['paired'])} items, fewer "
                f"than the {minimum_discordant_pairs(confidence)} an exact sign test needs to "
                "reach this level, so the interval cannot resolve a direction"
            ),
        )
    elif delta["lo"] >= -margin and delta["hi"] <= margin:
        result.update(
            resolution=POWER_RESOLUTION_FLAT,
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
            resolution=POWER_RESOLUTION_UNDECIDABLE,
            direction="none",
            reason=f"the paired interval crosses zero and the detectable-effect band; {size_note}",
        )
    return cast(LongContextPowerAnalysis, result)


def write_power_plan(plan: LongContextPowerAnalysis, path: Path) -> None:
    """Persist the declaration before any new model inference begins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_TARGET_POWER",
    "plan_from_artifact",
    "required_sample_size",
    "resolve_power_analysis",
    "write_power_plan",
]
