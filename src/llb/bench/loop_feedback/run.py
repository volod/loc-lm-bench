"""Paired completion, cost, and redirect analysis for repeat-feedback variants."""

from collections import Counter
from typing import cast

from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    REPEATED_NOOP,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.loop_feedback.outcomes import summarize_response_completion
from llb.bench.loop_policy.power import validate_repeat_power_design
from llb.bench.loop_policy.report import (
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
    run_seed: int | None = None,
) -> None:
    """Validate shared power constraints plus the current/localized feedback contract."""
    variants = feedback_variants_for_family(design, model_family)
    power_design = dict(design)
    power_design["repeat_feedback_variants"] = variants
    validate_repeat_power_design(power_design, tasks, cells=cells, model_family=model_family)
    if variants[0] != DEFAULT_REPEAT_FEEDBACK or len(set(variants)) != len(variants):
        raise ValueError("repeat feedback variants must be unique and start with current")
    if len(variants) < 2:
        raise ValueError("repeat feedback study needs current plus at least one candidate")
    declared_seeds = design.get("run_seeds")
    if declared_seeds is not None:
        seeds = cast(list[int], declared_seeds)
        if run_seed is None or run_seed not in seeds:
            raise ValueError(f"run_seed must be one of the predeclared seeds: {seeds}")


def feedback_variants_for_family(design: dict[str, object], model_family: str | None) -> list[str]:
    """Resolve the predeclared feedback cells for one family coordinate."""
    if design.get("study_kind") != "repeat_feedback_family_adaptation":
        return cast(list[str], design["repeat_feedback_variants"])
    roster = cast(list[dict[str, object]], design["roster"])
    row = next(
        (item for item in roster if item.get("model_family") == model_family),
        None,
    )
    if row is None:
        families = [item.get("model_family") for item in roster]
        raise ValueError(f"model_family must be one of the predeclared families: {families}")
    return [DEFAULT_REPEAT_FEEDBACK, cast(str, row["candidate_feedback_variant"])]


def _redirect_summary(
    report: LoopPolicyReport,
    tasks: list[AgenticTask],
) -> dict[str, object]:
    if len(report.rows) != len(tasks):
        raise ValueError("repeat-feedback rows do not match the task ledger")
    return summarize_response_completion(report.rows)


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


def _activation_gate(
    redirect: dict[str, object],
    *,
    task_count: int,
    minimum_rate: float,
    minimum_per_family: int,
) -> tuple[float, bool]:
    activation_rate = cast(int, redirect["activated_tasks"]) / task_count
    family_passed = all(
        cast(int, row["activated_tasks"]) >= minimum_per_family
        for row in cast(dict[str, dict[str, object]], redirect["by_family"]).values()
    )
    return activation_rate, activation_rate >= minimum_rate and family_passed


def analyze_repeat_feedback(
    design: dict[str, object],
    tasks: list[AgenticTask],
    reports: list[LoopPolicyReport],
    *,
    model_family: str | None,
    run_seed: int | None = None,
) -> dict[str, object]:
    """Compare every localized notice against the current English no-op notice."""
    current = next(
        report
        for report in reports
        if report.cell.policy.repeated_call == REPEATED_NOOP
        and report.cell.policy.repeat_feedback == DEFAULT_REPEAT_FEEDBACK
    )
    variants = feedback_variants_for_family(design, model_family)
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
    baseline_redirect = _redirect_summary(current, tasks)
    baseline_activation_rate, baseline_activation_passed = _activation_gate(
        baseline_redirect,
        task_count=len(tasks),
        minimum_rate=minimum_activation,
        minimum_per_family=minimum_family_activation,
    )
    analyses: dict[str, dict[str, object]] = {}
    for variant in variants[1:]:
        candidate = candidates[variant]
        completion = _paired(candidate, current, METRIC_COMPLETION, indexes)
        prompt = _paired(candidate, current, METRIC_PROMPT_TOKENS, indexes)
        wall = _paired(candidate, current, METRIC_WALL_CLOCK, indexes)
        redirect = _redirect_summary(candidate, tasks)
        activation_rate, activation_passed = _activation_gate(
            redirect,
            task_count=len(tasks),
            minimum_rate=minimum_activation,
            minimum_per_family=minimum_family_activation,
        )
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
        "run_seed": run_seed,
        "coverage_passed": True,
        "baseline_feedback_variant": DEFAULT_REPEAT_FEEDBACK,
        "task_family_counts": dict(sorted(Counter(task.family or "" for task in tasks).items())),
        "baseline": {
            "completion_rate": current.run.result.objective_score,
            "mean_total_model_input_tokens": current.metric_mean(METRIC_PROMPT_TOKENS),
            "mean_wall_clock_s": current.metric_mean(METRIC_WALL_CLOCK),
            "activation_rate": baseline_activation_rate,
            "activation_passed": baseline_activation_passed,
            "redirect": baseline_redirect,
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
