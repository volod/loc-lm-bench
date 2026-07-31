"""Readable reporting for the current-versus-localized repeat-feedback study."""

from typing import cast

from llb.bench.agentic.loop_policy import DEFAULT_REPEAT_FEEDBACK
from llb.bench.agentic_loop_policy_report import METRIC_COMPLETION, METRIC_PROMPT_TOKENS


def _delta_text(row: dict[str, object], metric: str) -> str:
    if metric == METRIC_COMPLETION:
        comparison = cast(dict[str, object], row["completion"])["paired"]
        delta = cast(dict[str, float], cast(dict[str, object], comparison)["delta"])
    else:
        gate = cast(dict[str, object], cast(dict[str, object], row["cost"])[metric])
        delta = cast(dict[str, float], gate["paired_delta"])
    return f"{delta['mean']:+.3f} [{delta['lo']:+.3f},{delta['hi']:+.3f}]"


def format_repeat_feedback_table(analysis: dict[str, object]) -> str:
    """Render redirect, completion, and cost decisions against the current notice."""
    header = (
        f"{'feedback':<10} {'response':>8} {'complete':>8} {'d(complete)':<23} "
        f"{'d(prompt)':<27} {'completion-gate':<15} {'cost-gate':<9} supports"
    )
    lines = [header, "-" * len(header)]
    baseline = cast(dict[str, object], analysis["baseline"])
    redirect = cast(dict[str, object], baseline["redirect"])
    lines.append(
        f"{DEFAULT_REPEAT_FEEDBACK:<10} {cast(float, redirect['response_rate']):>8.3f} "
        f"{cast(float, baseline['completion_rate']):>8.3f} {'-':<23} {'-':<27} "
        f"{'reference':<15} {'reference':<9} -"
    )
    for name, row in cast(dict[str, dict[str, object]], analysis["variants"]).items():
        response = cast(dict[str, object], row["redirect"])["response_rate"]
        completion = cast(dict[str, object], row["completion"])
        cost = cast(dict[str, object], row["cost"])
        lines.append(
            f"{name:<10} {cast(float, response):>8.3f} "
            f"{cast(float, row['completion_rate']):>8.3f} "
            f"{_delta_text(row, METRIC_COMPLETION):<23} "
            f"{_delta_text(row, METRIC_PROMPT_TOKENS):<27} "
            f"{str(completion['passed']).lower():<15} "
            f"{str(cost['passed']).lower():<9} "
            f"{str(row['supports_variant']).lower()}"
        )
    return "\n".join(lines)
