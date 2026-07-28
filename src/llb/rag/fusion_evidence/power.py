"""A priori power and realized sensitivity for any paired comparison lane.

The operator chooses the smallest material paired delta. An earlier item ledger supplies the
variance and discordance rate, so one pre-inference contract can state the item count needed by
both the normal-approximation MDE calculation and the exact-sign-test evidence floor.

This is sensitivity analysis, not post-hoc achieved power. After a run, the same arithmetic is
inverted to report the smallest delta the reached item set could resolve, and is repeated with the
run's own SD so a quieter reference set cannot make an underpowered run look complete.
"""

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from statistics import NormalDist
from typing import Any

from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs
from llb.rag.fusion_evidence.paired import (
    PairedComparison,
    compared_pairs,
    discordant_deltas,
    discordant_pairs,
)

POWER_METHOD = "paired-normal-approximation"
DEFAULT_TARGET_POWER = 0.80

DeltaSelector = Callable[[dict[str, Any]], list[float]]
PowerAnalysis = dict[str, Any]


def sample_sd(values: list[float]) -> float:
    """Sample SD of paired deltas."""
    if len(values) < 2:
        raise ValueError("the power reference needs at least two paired items")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _critical_value(alpha: float, target_power: float) -> float:
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    if not 0.5 < target_power < 1.0:
        raise ValueError("target power must be between 0.5 and 1")
    normal = NormalDist()
    return normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(target_power)


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
    critical = _critical_value(alpha, target_power)
    if sample_sd == 0.0:
        return 2
    return max(2, math.ceil((critical * sample_sd / minimum_detectable_delta) ** 2))


def resolvable_mde(sample_sd: float, n: int, *, alpha: float, target_power: float) -> float:
    """Smallest paired delta the reached `n` resolves under the declared approximation."""
    if sample_sd < 0.0:
        raise ValueError("sample SD must be non-negative")
    if n < 2:
        raise ValueError("resolvable MDE needs at least two paired items")
    return _critical_value(alpha, target_power) * sample_sd / math.sqrt(n)


def evidence_floor_n(deltas: list[float], confidence: float) -> int | None:
    """Item count that supplies enough discordant pairs for the reporting level."""
    discordant = discordant_deltas(deltas)
    if not deltas or discordant == 0:
        return None
    return math.ceil(minimum_discordant_pairs(confidence) * len(deltas) / discordant)


def _combined_floor(variance_n: int, evidence_n: int | None) -> tuple[int, str]:
    if evidence_n is None or variance_n > evidence_n:
        return variance_n, "variance"
    if evidence_n > variance_n:
        return evidence_n, "discordance"
    return variance_n, "both"


def plan_from_deltas(
    reference_artifact: Path,
    deltas: list[float],
    *,
    minimum_detectable_delta: float,
    target_power: float,
    confidence: float,
    planned_n: int,
    selector: dict[str, str],
) -> PowerAnalysis:
    """Build a current pre-inference declaration from an earlier paired item ledger."""
    sd = sample_sd(deltas)
    alpha = round(1.0 - confidence, 12)
    variance_n = required_sample_size(
        sd, minimum_detectable_delta, alpha=alpha, target_power=target_power
    )
    plan: PowerAnalysis = {
        "method": POWER_METHOD,
        "reference_artifact": str(reference_artifact),
        "reference_n": len(deltas),
        "reference_mean": sum(deltas) / len(deltas),
        "reference_sample_sd": sd,
        "minimum_detectable_delta": minimum_detectable_delta,
        "target_power": target_power,
        "alpha": alpha,
        "required_n": variance_n,
        "planned_n": planned_n,
        "target_reached": planned_n >= variance_n,
    }
    evidence_n = evidence_floor_n(deltas, confidence)
    required_n, binding = _combined_floor(variance_n, evidence_n)
    plan.update(
        selector=selector,
        variance_required_n=variance_n,
        evidence_floor_n=evidence_n,
        binding_floor=binding,
        required_n=required_n,
        target_reached=planned_n >= required_n,
    )
    return plan


def plan_from_artifact(
    reference_artifact: Path,
    select_deltas: DeltaSelector,
    **kwargs: Any,
) -> PowerAnalysis:
    """Read an earlier artifact and price the selected paired delta."""
    try:
        payload = json.loads(reference_artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read power reference {reference_artifact}: {exc}") from exc
    return plan_from_deltas(reference_artifact, select_deltas(payload), **kwargs)


def resolve_power_analysis(
    plan: PowerAnalysis,
    deltas: list[float],
    paired: PairedComparison,
    *,
    candidate: str,
    baseline: str,
) -> PowerAnalysis:
    """Re-check sensitivity with realized variance and attach the reached reading."""
    result = dict(plan)
    confidence = 1.0 - plan["alpha"]
    sd = sample_sd(deltas)
    variance_n = required_sample_size(
        sd,
        plan["minimum_detectable_delta"],
        alpha=plan["alpha"],
        target_power=plan["target_power"],
    )
    evidence_n = evidence_floor_n(deltas, confidence)
    required_n, binding = _combined_floor(variance_n, evidence_n)
    reached = len(deltas) >= required_n
    result.update(
        planned_target_reached=plan["target_reached"],
        realized_n=len(deltas),
        realized_mean=sum(deltas) / len(deltas),
        realized_sample_sd=sd,
        realized_required_n=required_n,
        realized_evidence_floor_n=evidence_n,
        realized_binding_floor=binding,
        resolvable_mde=resolvable_mde(
            sd, len(deltas), alpha=plan["alpha"], target_power=plan["target_power"]
        ),
        realized_sd_exceeds_plan=sd > plan["reference_sample_sd"],
        target_reached=reached,
    )
    interval = paired["delta"]
    margin = plan["minimum_detectable_delta"]
    enough_discordance = discordant_pairs(paired) >= minimum_discordant_pairs(confidence)
    if interval["lo"] > 0.0 and enough_discordance:
        resolution, direction, reason = (
            "separated",
            candidate,
            "the paired interval is wholly above zero",
        )
    elif interval["hi"] < 0.0 and enough_discordance:
        resolution, direction, reason = (
            "separated",
            baseline,
            "the paired interval is wholly below zero",
        )
    elif interval["lo"] >= -margin and interval["hi"] <= margin:
        resolution, direction, reason = (
            "flat",
            "neither",
            "the paired interval lies wholly inside the predeclared detectable-effect band",
        )
    else:
        resolution, direction = "undecidable", "none"
        if not enough_discordance:
            reason = (
                f"only {discordant_pairs(paired)} of {compared_pairs(paired)} paired items differ; "
                f"the realized discordance floor is {evidence_n or 'not estimable'} items"
            )
        elif not reached:
            reason = (
                "the run's realized variance requires "
                f"{required_n} items, above the {len(deltas)} reached"
            )
        else:
            reason = "the paired interval crosses zero and the detectable-effect band"
    result.update(resolution=resolution, direction=direction, reason=reason)
    return result


def write_power_plan(plan: Mapping[str, Any], path: Path) -> None:
    """Persist the declaration before the lane performs new inference or retrieval."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_TARGET_POWER",
    "POWER_METHOD",
    "PowerAnalysis",
    "evidence_floor_n",
    "plan_from_artifact",
    "plan_from_deltas",
    "required_sample_size",
    "resolvable_mde",
    "resolve_power_analysis",
    "sample_sd",
    "write_power_plan",
]
