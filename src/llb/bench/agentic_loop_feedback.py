"""Paired completion, cost, and redirect analysis for repeat-feedback variants."""

from collections import Counter
from typing import cast

from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    REPEATED_NOOP,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_loop_policy_power import validate_repeat_power_design
from llb.bench.agentic_loop_policy_report import (
    METRIC_COMPLETION,
    METRIC_PROMPT_TOKENS,
    METRIC_WALL_CLOCK,
    LoopPolicyCell,
    LoopPolicyReport,
)
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison, reading_of
from llb.rag.fusion_evidence.stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_index_sets,
)


def validate_repeat_feedback_design(
    design: dict[str, object],
    tasks: list[AgenticTask],
    *,
    cells: list[LoopPolicyCell],
    model_family: str | None,
) -> None:
    """Validate shared power constraints plus the current/localized feedback contract."""
    validate_repeat_power_design(design, tasks, cells=cells, model_family=model_family)
    variants = cast(list[str], design["repeat_feedback_variants"])
    if variants[0] != DEFAULT_REPEAT_FEEDBACK or len(set(variants)) != len(variants):
        raise ValueError("repeat feedback variants must be unique and start with current")
    if len(variants) < 2:
        raise ValueError("repeat feedback study needs current plus at least one candidate")


def _redirect_summary(
    report: LoopPolicyReport,
    tasks: list[AgenticTask],
) -> dict[str, object]:
    active = [episode.n_repeated_noops > 0 for episode in report.episodes]
    redirected = [episode.repeat_feedback_redirected for episode in report.episodes]
    by_family: dict[str, dict[str, object]] = {}
    for family in sorted({task.family or "" for task in tasks}):
        indexes = [index for index, task in enumerate(tasks) if task.family == family]
        activated = sum(active[index] for index in indexes)
        responses = sum(redirected[index] for index in indexes)
        by_family[family] = {
            "tasks": len(indexes),
            "activated_tasks": activated,
            "redirected_tasks": responses,
            "response_rate": responses / activated if activated else 0.0,
        }
    activated = sum(active)
    responses = sum(redirected)
    return {
        "tasks": len(tasks),
        "activated_tasks": activated,
        "redirected_tasks": responses,
        "response_rate": responses / activated if activated else 0.0,
        "by_family": by_family,
    }


def _paired(
    candidate: LoopPolicyReport,
    baseline: LoopPolicyReport,
    metric: str,
    indexes: list[list[int]],
) -> PairedComparison:
    return paired_comparison(candidate.vector(metric), baseline.vector(metric), indexes)


def _cost_gate(
    comparison: PairedComparison,
    baseline_mean: float,
    relative_limit: float,
) -> dict[str, object]:
    allowed_delta = baseline_mean * relative_limit
    delta = cast(dict[str, float], comparison["delta"])
    return {
        "relative_increase_limit": relative_limit,
        "baseline_mean": baseline_mean,
        "allowed_delta": allowed_delta,
        "paired_delta": delta,
        "passed": delta["hi"] <= allowed_delta,
    }


def analyze_repeat_feedback(
    design: dict[str, object],
    tasks: list[AgenticTask],
    reports: list[LoopPolicyReport],
    *,
    model_family: str | None,
) -> dict[str, object]:
    """Compare every localized notice against the current English no-op notice."""
    current = next(
        report
        for report in reports
        if report.cell.policy.repeated_call == REPEATED_NOOP
        and report.cell.policy.repeat_feedback == DEFAULT_REPEAT_FEEDBACK
    )
    variants = cast(list[str], design["repeat_feedback_variants"])
    candidates = {
        report.cell.policy.repeat_feedback: report
        for report in reports
        if report.cell.policy.repeated_call == REPEATED_NOOP
        and report.cell.policy.repeat_feedback != DEFAULT_REPEAT_FEEDBACK
    }
    indexes = bootstrap_index_sets(len(tasks), DEFAULT_RESAMPLES, DEFAULT_SEED)
    mde = float(cast(float, design["minimum_detectable_completion_gain"]))
    minimum_activation = float(cast(float, design["minimum_activation_rate"]))
    minimum_family_activation = int(cast(int, design["minimum_activated_tasks_per_family"]))
    cost_limits = cast(dict[str, float], design["maximum_relative_cost_increase"])
    analyses: dict[str, dict[str, object]] = {}
    for variant in variants[1:]:
        candidate = candidates[variant]
        completion = _paired(candidate, current, METRIC_COMPLETION, indexes)
        prompt = _paired(candidate, current, METRIC_PROMPT_TOKENS, indexes)
        wall = _paired(candidate, current, METRIC_WALL_CLOCK, indexes)
        redirect = _redirect_summary(candidate, tasks)
        family_activation_passed = all(
            cast(int, row["activated_tasks"]) >= minimum_family_activation
            for row in cast(dict[str, dict[str, object]], redirect["by_family"]).values()
        )
        activation_rate = cast(int, redirect["activated_tasks"]) / len(tasks)
        activation_passed = activation_rate >= minimum_activation and family_activation_passed
        completion_delta = cast(dict[str, float], completion["delta"])
        completion_reading = reading_of(completion)
        completion_passed = completion_delta["mean"] >= mde and completion_reading == "separated"
        prompt_gate = _cost_gate(
            prompt,
            current.metric_mean(METRIC_PROMPT_TOKENS),
            float(cost_limits[METRIC_PROMPT_TOKENS]),
        )
        wall_gate = _cost_gate(
            wall,
            current.metric_mean(METRIC_WALL_CLOCK),
            float(cost_limits[METRIC_WALL_CLOCK]),
        )
        supports = bool(
            activation_passed
            and completion_passed
            and prompt_gate["passed"]
            and wall_gate["passed"]
        )
        analyses[variant] = {
            "completion_rate": candidate.run.result.objective_score,
            "mean_total_model_input_tokens": candidate.metric_mean(METRIC_PROMPT_TOKENS),
            "mean_wall_clock_s": candidate.metric_mean(METRIC_WALL_CLOCK),
            "activation_rate": activation_rate,
            "activation_passed": activation_passed,
            "redirect": redirect,
            "completion": {
                "minimum_detectable_gain": mde,
                "paired": completion,
                "reading": completion_reading,
                "passed": completion_passed,
            },
            "cost": {
                METRIC_PROMPT_TOKENS: prompt_gate,
                METRIC_WALL_CLOCK: wall_gate,
                "passed": bool(prompt_gate["passed"] and wall_gate["passed"]),
            },
            "supports_variant": supports,
        }
    supported = [name for name, row in analyses.items() if row["supports_variant"]]
    recommended = min(
        supported,
        key=lambda name: (
            -candidates[name].run.result.objective_score,
            candidates[name].metric_mean(METRIC_PROMPT_TOKENS),
            candidates[name].metric_mean(METRIC_WALL_CLOCK),
        ),
        default=DEFAULT_REPEAT_FEEDBACK,
    )
    return {
        "study_id": design["study_id"],
        "model_family": model_family,
        "coverage_passed": True,
        "baseline_feedback_variant": DEFAULT_REPEAT_FEEDBACK,
        "task_family_counts": dict(sorted(Counter(task.family or "" for task in tasks).items())),
        "baseline": {
            "completion_rate": current.run.result.objective_score,
            "mean_total_model_input_tokens": current.metric_mean(METRIC_PROMPT_TOKENS),
            "mean_wall_clock_s": current.metric_mean(METRIC_WALL_CLOCK),
            "redirect": _redirect_summary(current, tasks),
        },
        "variants": analyses,
        "recommended_feedback_variant": recommended,
        "supports_localized_feedback": bool(supported),
        "reason": (
            "a localized variant clears activation, material completion, and paired cost gates"
            if supported
            else "no localized variant clears every prospective gate against the current notice"
        ),
    }


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


__all__ = [
    "analyze_repeat_feedback",
    "format_repeat_feedback_table",
    "validate_repeat_feedback_design",
]
