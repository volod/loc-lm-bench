"""Paired-power artifact selector for the embedder bake-off."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llb.rag.embedding_bakeoff.models import BakeoffReport
from llb.rag.embedding_bakeoff.uncertainty import METRICS
from llb.rag.fusion_evidence.power import (
    PowerAnalysis,
    plan_from_artifact,
    resolve_power_analysis,
    write_power_plan,
)


def _selector(candidate: str, baseline: str, metric: str) -> dict[str, str]:
    return {
        "lane": "compare-embeddings",
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "population": "all",
    }


def paired_deltas(payload: Mapping[str, Any], candidate: str, metric: str) -> list[float]:
    """Read candidate-minus-baseline values from the persisted per-item metric ledger."""
    uncertainty = payload.get("uncertainty", {})
    baseline = uncertainty.get("baseline")
    items = payload.get("paired_items")
    if not baseline or not isinstance(items, list):
        raise ValueError(
            "embedder power reference needs uncertainty.baseline and a paired_items ledger"
        )
    deltas: list[float] = []
    for item in items:
        models = item.get("models", {})
        try:
            deltas.append(float(models[candidate][metric]) - float(models[baseline][metric]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"embedder power reference lacks {candidate!r} versus {baseline!r} {metric!r}"
            ) from exc
    return deltas


def plan_embedding_power(
    reference: Path,
    *,
    candidate: str,
    metric: str,
    minimum_detectable_delta: float,
    target_power: float,
    confidence: float,
    planned_n: int,
) -> PowerAnalysis:
    """Price one declared embedder delta from an earlier bake-off artifact."""

    def select(payload: dict[str, Any]) -> list[float]:
        return paired_deltas(payload, candidate, metric)

    import json

    try:
        payload = json.loads(reference.read_text(encoding="utf-8"))
        baseline = str(payload["uncertainty"]["baseline"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read embedder power reference {reference}: {exc}") from exc
    return plan_from_artifact(
        reference,
        select,
        minimum_detectable_delta=minimum_detectable_delta,
        target_power=target_power,
        confidence=confidence,
        planned_n=planned_n,
        selector=_selector(candidate, baseline, metric),
    )


def resolve_embedding_power(report: BakeoffReport, plan: PowerAnalysis) -> PowerAnalysis:
    """Attach realized sensitivity and resolution to a completed bake-off report."""
    candidate = str(plan["selector"]["candidate"])
    metric = str(plan["selector"]["metric"])
    baseline = report.get("uncertainty", {}).get("baseline")
    if baseline != plan["selector"]["baseline"]:
        raise ValueError(
            f"power reference baseline {plan['selector']['baseline']!r} does not match "
            f"scored baseline {baseline!r}"
        )
    row = next(
        (entry for entry in report.get("candidates", []) if entry.get("model") == candidate), None
    )
    if not baseline or row is None:
        raise ValueError(f"power-selected embedder {candidate!r} was not scored with a baseline")
    try:
        paired = row["paired_vs_baseline"]["metrics"][metric]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"power-selected embedder metric {metric!r} was not scored") from exc
    return resolve_power_analysis(
        plan,
        paired_deltas(report, candidate, metric),
        paired,
        candidate=candidate,
        baseline=str(baseline),
    )


def prepare_embedding_power(
    reference: Path | None,
    *,
    candidate: str | None,
    metric: str,
    minimum_detectable_delta: float | None,
    target_power: float,
    confidence: float,
    planned_n: int,
    plan_path: Path,
) -> PowerAnalysis | None:
    """Validate optional CLI fields and persist a complete declaration, or return None."""
    if reference is None and minimum_detectable_delta is None:
        return None
    if reference is None or minimum_detectable_delta is None:
        raise ValueError(
            "power planning needs both --power-reference and --minimum-detectable-delta"
        )
    if candidate is None:
        raise ValueError("power planning needs --power-candidate")
    if metric not in METRICS:
        raise ValueError(f"--power-metric must be one of {','.join(METRICS)}")
    plan = plan_embedding_power(
        reference,
        candidate=candidate,
        metric=metric,
        minimum_detectable_delta=minimum_detectable_delta,
        target_power=target_power,
        confidence=confidence,
        planned_n=planned_n,
    )
    write_power_plan(plan, plan_path)
    return plan


__all__ = [
    "paired_deltas",
    "plan_embedding_power",
    "prepare_embedding_power",
    "resolve_embedding_power",
]
