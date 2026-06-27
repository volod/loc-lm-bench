"""Guarded composite scoring for the M5 category suite.

The per-category boards deliberately stay isolated by Tier. This module builds a separate M5
composite headline only after each required category contributes a verified, CI-capable objective
series for the same model. The weights mirror the spec taxonomy's M5-category proportions and are
renormalized over the M5-only subset (RAG, chat-period, reliability, and efficiency stay outside
this first composite layer).
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llb.contracts import JsonObject
from llb.scoring.aggregate import (
    TIER_AGENTIC,
    TIER_SECURITY,
    TIER_STRUCTURED,
    TIER_SUMMARIZATION,
    TIER_TEXT_ANALYSIS,
    TIER_TOOLING,
    ModelResult,
)

M5_COMPOSITE_RAW_WEIGHTS: dict[str, float] = {
    TIER_TEXT_ANALYSIS: 20.0,
    TIER_SUMMARIZATION: 10.0,
    TIER_STRUCTURED: 10.0,
    TIER_SECURITY: 10.0,
    TIER_AGENTIC: 10.0,
    TIER_TOOLING: 5.0,
}
M5_COMPOSITE_REQUIRED_TIERS: tuple[str, ...] = tuple(M5_COMPOSITE_RAW_WEIGHTS)
_DEFAULT_RESAMPLES = 1000
_DEFAULT_SEED = 0
_ROUND_DIGITS = 4


@dataclass(frozen=True)
class CompositeComponent:
    """One category result plus the data-gate metadata needed before headline use."""

    result: ModelResult
    data_verified: bool = False
    verification_ref: str | None = None
    verification_error: str | None = None


@dataclass(frozen=True)
class CompositeIssue:
    """Why a model is blocked from the composite headline."""

    model: str
    reason: str
    tier: str | None = None


def normalized_m5_weights(
    raw_weights: Mapping[str, float] = M5_COMPOSITE_RAW_WEIGHTS,
) -> dict[str, float]:
    """Normalize the M5 raw weights to sum to 1.0."""
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("composite weights must have a positive sum")
    return {tier: weight / total for tier, weight in raw_weights.items()}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[float], q: float) -> float:
    index = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values))))
    return sorted_values[index]


def _composite_ci(
    components: Sequence[CompositeComponent],
    weights: Mapping[str, float],
    *,
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, float] | None:
    """Bootstrap the weighted category-mean composite from each component's case series."""
    if any(len(component.result.case_objectives) < 2 for component in components):
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for component in components:
            values = component.result.case_objectives
            sampled = [values[rng.randrange(len(values))] for _ in range(len(values))]
            total += weights[component.result.tier] * _mean(sampled)
        means.append(total)
    means.sort()
    return (
        round(_percentile(means, 0.025), _ROUND_DIGITS),
        round(_percentile(means, 0.975), _ROUND_DIGITS),
    )


def _models(components_by_tier: Mapping[str, Sequence[CompositeComponent]]) -> list[str]:
    return sorted(
        {component.result.model for values in components_by_tier.values() for component in values}
    )


def _by_model(components: Sequence[CompositeComponent]) -> dict[str, CompositeComponent]:
    return {component.result.model: component for component in components}


def _component_issues(
    model: str,
    component: CompositeComponent | None,
    tier: str,
    *,
    require_verified: bool,
    require_ci: bool,
) -> list[CompositeIssue]:
    if component is None:
        return [CompositeIssue(model=model, tier=tier, reason="missing required tier")]
    issues: list[CompositeIssue] = []
    if require_verified and not component.data_verified:
        issues.append(
            CompositeIssue(model=model, tier=tier, reason="category data is not verified")
        )
    elif require_verified and component.verification_error:
        issues.append(
            CompositeIssue(
                model=model,
                tier=tier,
                reason=f"verification ref invalid: {component.verification_error}",
            )
        )
    if require_ci and len(component.result.case_objectives) < 2:
        issues.append(
            CompositeIssue(model=model, tier=tier, reason="category lacks per-case CI series")
        )
    return issues


def build_m5_composite_rows(
    components_by_tier: Mapping[str, Sequence[CompositeComponent]],
    *,
    require_verified: bool = True,
    require_ci: bool = True,
    raw_weights: Mapping[str, float] = M5_COMPOSITE_RAW_WEIGHTS,
) -> tuple[list[JsonObject], list[CompositeIssue]]:
    """Build ranked M5 composite rows plus blocking issues.

    A model receives a row only when it has every required tier and satisfies the requested gates.
    The row is not a replacement for the per-tier boards; it is a separate headline over the
    already-normalized category objective scores.
    """
    weights = normalized_m5_weights(raw_weights)
    indexed = {tier: _by_model(components_by_tier.get(tier, ())) for tier in weights}
    rows: list[JsonObject] = []
    issues: list[CompositeIssue] = []
    for model in _models(components_by_tier):
        components: list[CompositeComponent] = []
        model_issues: list[CompositeIssue] = []
        for tier in weights:
            component = indexed[tier].get(model)
            model_issues.extend(
                _component_issues(
                    model,
                    component,
                    tier,
                    require_verified=require_verified,
                    require_ci=require_ci,
                )
            )
            if component is not None:
                components.append(component)
        if model_issues:
            issues.extend(model_issues)
            continue
        score = sum(
            weights[component.result.tier] * component.result.objective_score
            for component in components
        )
        ci = _composite_ci(components, weights) if require_ci else None
        row: JsonObject = {
            "rank": None,
            "model": model,
            "score": round(score, _ROUND_DIGITS),
            "avg_reliability": round(
                _mean([c.result.reliability for c in components]), _ROUND_DIGITS
            ),
            "n_cases": sum(component.result.n_cases for component in components),
            "unresolved": False,
        }
        if ci is not None:
            row["score_ci"] = ci
        for component in components:
            tier = component.result.tier
            row[tier] = round(component.result.objective_score, _ROUND_DIGITS)
            row[f"{tier}_n"] = component.result.n_cases
            if component.verification_ref:
                row[f"{tier}_verification"] = component.verification_ref
        rows.append(row)

    rows.sort(
        key=lambda row: (-float(row["score"]), -float(row["avg_reliability"]), str(row["model"]))
    )
    previous_ci: tuple[float, float] | None = None
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        ci = row.get("score_ci")
        if isinstance(ci, tuple):
            row["unresolved"] = bool(previous_ci and ci[1] >= previous_ci[0])
            previous_ci = ci
        else:
            row["unresolved"] = False
    return rows, issues


def format_composite_issues(issues: Sequence[CompositeIssue], *, limit: int = 12) -> str:
    """ASCII summary of why the composite headline is blocked."""
    if not issues:
        return ""
    lines = ["M5 composite headline is blocked:"]
    for issue in issues[:limit]:
        tier = f" {issue.tier}" if issue.tier else ""
        lines.append(f"- {issue.model}{tier}: {issue.reason}")
    if len(issues) > limit:
        lines.append(f"- ... {len(issues) - limit} more")
    return "\n".join(lines)


def format_composite_rows(rows: Sequence[JsonObject]) -> str:
    """Render composite rows as an ASCII table."""
    headers = ["rank", "model", "score", "ci", "avg_reliability", "n_cases", "unresolved"]

    def cell(row: JsonObject, key: str) -> str:
        ci = row.get("score_ci")
        mapping = {
            "rank": str(row.get("rank", "-")),
            "model": str(row["model"]),
            "score": f"{float(row['score']):.3f}",
            "ci": "-" if ci is None else f"[{ci[0]:.2f},{ci[1]:.2f}]",
            "avg_reliability": f"{float(row['avg_reliability']):.3f}",
            "n_cases": str(row["n_cases"]),
            "unresolved": "yes" if row["unresolved"] else "no",
        }
        return mapping[key]

    table = [[cell(row, header) for header in headers] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in table)) if table else len(header)
        for i, header in enumerate(headers)
    ]
    out = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in table:
        out.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(out)
