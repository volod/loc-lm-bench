"""Paired-power artifact selector for the graph-vector fusion evidence lane."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llb.rag.fusion_evidence.models import METRICS, FusionEvidenceReport
from llb.rag.fusion_evidence.power import (
    PowerAnalysis,
    plan_from_artifact,
    resolve_power_analysis,
    write_power_plan,
)


def paired_deltas(payload: Mapping[str, Any], row: str, metric: str) -> list[float]:
    """Read one focus-slice row-minus-baseline delta from the persisted item ledger."""
    baseline = payload.get("baseline")
    items = payload.get("focus_items")
    if not baseline or not isinstance(items, list):
        raise ValueError("fusion power reference needs baseline and focus_items")
    deltas: list[float] = []
    for item in items:
        rows = item.get("rows", {})
        try:
            deltas.append(float(rows[row][metric]) - float(rows[baseline][metric]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"fusion power reference lacks row {row!r} versus {baseline!r} metric {metric!r}"
            ) from exc
    return deltas


def plan_fusion_power(
    reference: Path,
    *,
    row: str,
    metric: str,
    minimum_detectable_delta: float,
    target_power: float,
    confidence: float,
    planned_n: int,
) -> PowerAnalysis:
    """Price one declared focus-slice fusion delta from an earlier comparison."""
    try:
        payload = json.loads(reference.read_text(encoding="utf-8"))
        baseline = str(payload["baseline"])
        population = str(payload["focus_slice"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read fusion power reference {reference}: {exc}") from exc
    if row == baseline:
        raise ValueError("fusion power row must differ from the baseline")

    def select(artifact: dict[str, Any]) -> list[float]:
        return paired_deltas(artifact, row, metric)

    return plan_from_artifact(
        reference,
        select,
        minimum_detectable_delta=minimum_detectable_delta,
        target_power=target_power,
        confidence=confidence,
        planned_n=planned_n,
        selector={
            "lane": "compare-graph-fusion",
            "candidate": row,
            "baseline": baseline,
            "metric": metric,
            "population": population,
        },
    )


def resolve_fusion_power(report: FusionEvidenceReport, plan: PowerAnalysis) -> PowerAnalysis:
    """Attach realized sensitivity and resolution to a completed fusion report."""
    row = str(plan["selector"]["candidate"])
    metric = str(plan["selector"]["metric"])
    baseline = str(report["baseline"])
    focus = str(report["focus_slice"])
    if baseline != plan["selector"]["baseline"] or focus != plan["selector"]["population"]:
        raise ValueError(
            "the scored fusion baseline or focus slice differs from the power selector"
        )
    try:
        paired = report["rows"][row]["slices"][focus]["paired_vs_baseline"][metric]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"power-selected fusion row {row!r} metric {metric!r} was not scored on {focus!r}"
        ) from exc
    return resolve_power_analysis(
        plan,
        paired_deltas(report, row, metric),
        paired,
        candidate=row,
        baseline=baseline,
    )


def prepare_fusion_power(
    reference: Path | None,
    *,
    row: str | None,
    metric: str,
    minimum_detectable_delta: float | None,
    target_power: float,
    confidence: float,
    planned_n: int,
    focus_slice: str,
    plan_path: Path,
) -> PowerAnalysis | None:
    """Validate optional CLI fields and persist a focus-slice declaration, or return None."""
    if reference is None and minimum_detectable_delta is None:
        return None
    if reference is None or minimum_detectable_delta is None:
        raise ValueError(
            "power planning needs both --power-reference and --minimum-detectable-delta"
        )
    if row is None:
        raise ValueError("power planning needs --power-row")
    if metric not in METRICS:
        raise ValueError(f"--power-metric must be one of {','.join(METRICS)}")
    try:
        reference_focus = json.loads(reference.read_text(encoding="utf-8")).get("focus_slice")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read fusion power reference {reference}: {exc}") from exc
    if reference_focus != focus_slice:
        raise ValueError(
            f"power reference population {reference_focus!r} does not match selected focus "
            f"slice {focus_slice!r}"
        )
    plan = plan_fusion_power(
        reference,
        row=row,
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
    "plan_fusion_power",
    "prepare_fusion_power",
    "resolve_fusion_power",
]
