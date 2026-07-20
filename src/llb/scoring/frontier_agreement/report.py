"""Render the frontier-judge agreement report the human signs off on."""

from typing import Any

from llb.core.contracts.judging import CalibrationResult
from llb.scoring.frontier_agreement.agreement import (
    AGREEMENT_METRICS,
    MEAN_METRIC,
    ProviderAgreement,
)

MISSING = "n/a"


def _rho_cell(result: CalibrationResult | None) -> str:
    if result is None:
        return MISSING
    return f"{result['rho']:.3f} [{result['ci_low']:.3f},{result['ci_high']:.3f}] n={result['n']}"


def _usd(value: float | None) -> str:
    return MISSING if value is None else f"${value:.6f}".rstrip("0").rstrip(".")


def _cap(value: float | None) -> str:
    return MISSING if value is None else f"${value:.2f}"


def _agreement_table(agreements: list[ProviderAgreement], attr: str, title: str) -> list[str]:
    lines = [f"### {title}", "", "| provider | model | " + " | ".join(AGREEMENT_METRICS) + " |"]
    lines.append("| --- | --- | " + " | ".join(["---"] * len(AGREEMENT_METRICS)) + " |")
    for item in agreements:
        block: dict[str, CalibrationResult | None] = getattr(item, attr)
        cells = " | ".join(_rho_cell(block[metric]) for metric in AGREEMENT_METRICS)
        lines.append(f"| {item.provider} | `{item.model}` | {cells} |")
    lines.append("")
    return lines


def _cost_table(agreements: list[ProviderAgreement]) -> list[str]:
    lines = [
        "### Cost per item and cap math",
        "",
        "| provider | model | calls | total | per item | recommended cap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in agreements:
        cost = item.cost
        lines.append(
            f"| {item.provider} | `{item.model}` | {cost['calls']} | "
            f"{_usd(cost['cost_usd'])} | {_usd(cost['cost_per_item_usd'])} | "
            f"{_cap(cost['recommended_cap_usd'])} |"
        )
    lines.append("")
    unpriced = [item.model for item in agreements if not item.cost["priced"]]
    if unpriced:
        lines.append(
            "Unpriced by litellm (cost recorded as 0, cap not recommended): "
            + ", ".join(f"`{model}`" for model in unpriced)
            + "."
        )
        lines.append("")
    return lines


def _decision_table(agreements: list[ProviderAgreement]) -> list[str]:
    lines = [
        "### Human decision (operator fills this in)",
        "",
        "| provider | model | rho vs human | machine recommendation | human decision |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in agreements:
        headline = item.headline
        rho = MISSING if headline is None else f"{headline['rho']:.3f}"
        lines.append(
            f"| {item.provider} | `{item.model}` | {rho} | {item.recommendation} | pending |"
        )
    lines.extend(
        [
            "",
            "A provider is authorized for autonomous gating only when the operator records an",
            "explicit accept here. `trusted` above means the measured rho cleared the threshold,",
            "not that the provider has been authorized.",
            "",
        ]
    )
    return lines


def render_report(payload: dict[str, Any], agreements: list[ProviderAgreement]) -> str:
    """Markdown report: agreement against both references, cost, and the sign-off table."""
    threshold = payload["threshold"]
    lines = [
        "# Frontier judge agreement and cost",
        "",
        f"- run: `{payload['run']}`",
        f"- worksheet: `{payload['worksheet']}`",
        f"- gold set: `{payload['goldset'] or MISSING}`",
        f"- items judged: {payload['n_items']}",
        f"- trust threshold: rho >= {threshold}",
        f"- headline metric: `{MEAN_METRIC}` (the scalar a scored run records per case)",
        "",
        "Ratings are compared by rank (Spearman), so the human 1..5 scale and the judge 0..1",
        "scale need no rescaling. Intervals are bootstrap 95% CIs.",
        "",
    ]
    lines += _agreement_table(agreements, "vs_human", "Frontier vs human rating")
    lines += _agreement_table(agreements, "vs_local", "Frontier vs local judge rating")
    lines += _cost_table(agreements)
    lines += _decision_table(agreements)
    return "\n".join(lines) + "\n"
