"""Row construction for the guarded category composite."""

from collections.abc import Mapping, Sequence

from llb.contracts import JsonObject
from llb.scoring.composite_stats import ROUND_DIGITS, bootstrap_weighted_mean_ci, mean
from llb.scoring.composite_types import (
    CATEGORY_COMPOSITE_RAW_WEIGHTS,
    CompositeComponent,
    CompositeIssue,
)


def normalized_composite_weights(
    raw_weights: Mapping[str, float] = CATEGORY_COMPOSITE_RAW_WEIGHTS,
) -> dict[str, float]:
    """Normalize raw category weights to sum to 1.0."""
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("composite weights must have a positive sum")
    return {tier: weight / total for tier, weight in raw_weights.items()}


def build_category_composite_rows(
    components_by_tier: Mapping[str, Sequence[CompositeComponent]],
    *,
    require_verified: bool = True,
    require_ci: bool = True,
    raw_weights: Mapping[str, float] = CATEGORY_COMPOSITE_RAW_WEIGHTS,
) -> tuple[list[JsonObject], list[CompositeIssue]]:
    """Build ranked category-suite composite rows plus blocking issues."""
    weights = normalized_composite_weights(raw_weights)
    tiers = tuple(weights)
    indexed = _indexed_components(components_by_tier, tiers)
    rows: list[JsonObject] = []
    issues: list[CompositeIssue] = []
    for model in _models(components_by_tier):
        components, model_issues = _model_components(
            model,
            tiers,
            indexed,
            require_verified=require_verified,
            require_ci=require_ci,
        )
        if model_issues:
            issues.extend(model_issues)
        else:
            rows.append(_composite_row(model, components, weights, require_ci=require_ci))
    _rank_composite_rows(rows)
    return rows, issues


def _models(components_by_tier: Mapping[str, Sequence[CompositeComponent]]) -> list[str]:
    return sorted(
        {component.result.model for values in components_by_tier.values() for component in values}
    )


def _by_model(components: Sequence[CompositeComponent]) -> dict[str, CompositeComponent]:
    return {component.result.model: component for component in components}


def _indexed_components(
    components_by_tier: Mapping[str, Sequence[CompositeComponent]],
    tiers: Sequence[str],
) -> dict[str, dict[str, CompositeComponent]]:
    return {tier: _by_model(components_by_tier.get(tier, ())) for tier in tiers}


def _model_components(
    model: str,
    tiers: Sequence[str],
    indexed: Mapping[str, Mapping[str, CompositeComponent]],
    *,
    require_verified: bool,
    require_ci: bool,
) -> tuple[list[CompositeComponent], list[CompositeIssue]]:
    components: list[CompositeComponent] = []
    issues: list[CompositeIssue] = []
    for tier in tiers:
        component = indexed[tier].get(model)
        issues.extend(
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
    return components, issues


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


def _composite_ci(
    components: Sequence[CompositeComponent],
    weights: Mapping[str, float],
) -> tuple[float, float] | None:
    return bootstrap_weighted_mean_ci(
        [(component.result.tier, component.result.case_objectives) for component in components],
        weights,
    )


def _composite_score(
    components: Sequence[CompositeComponent],
    weights: Mapping[str, float],
) -> float:
    return sum(
        weights[component.result.tier] * component.result.objective_score
        for component in components
    )


def _component_columns(row: JsonObject, components: Sequence[CompositeComponent]) -> None:
    for component in components:
        tier = component.result.tier
        row[tier] = round(component.result.objective_score, ROUND_DIGITS)
        row[f"{tier}_n"] = component.result.n_cases
        if component.verification_ref:
            row[f"{tier}_verification"] = component.verification_ref


def _composite_row(
    model: str,
    components: Sequence[CompositeComponent],
    weights: Mapping[str, float],
    *,
    require_ci: bool,
) -> JsonObject:
    row: JsonObject = {
        "rank": None,
        "model": model,
        "score": round(_composite_score(components, weights), ROUND_DIGITS),
        "avg_reliability": round(mean([c.result.reliability for c in components]), ROUND_DIGITS),
        "n_cases": sum(component.result.n_cases for component in components),
        "unresolved": False,
    }
    ci = _composite_ci(components, weights) if require_ci else None
    if ci is not None:
        row["score_ci"] = ci
    _component_columns(row, components)
    return row


def _rank_composite_rows(rows: list[JsonObject]) -> None:
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
