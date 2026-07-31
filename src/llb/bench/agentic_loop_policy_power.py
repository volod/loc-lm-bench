"""Prospective design and realized gates for the focused repeat/no-op comparison."""

import json
from collections import Counter
from pathlib import Path
from typing import cast

from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    REPEATED_ALLOW,
    REPEATED_NOOP,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_loop_policy_report import (
    BASELINE_MAX_STEPS,
    METRIC_COMPLETION,
    METRIC_PROMPT_TOKENS,
    METRIC_WALL_CLOCK,
    LoopPolicyCell,
    LoopPolicyReport,
)
from llb.rag.fusion_evidence.paired import reading_of

DESIGN_SCHEMA_VERSION = 1


def load_repeat_power_design(path: Path | str) -> dict[str, object]:
    """Load one committed prospective design without mutating it after inference."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read repeat-power design {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("repeat-power design must be a JSON object")
    return cast(dict[str, object], raw)


def _family_counts(tasks: list[AgenticTask]) -> Counter[str]:
    return Counter(task.family or "" for task in tasks)


def _duplicate_payloads(tasks: list[AgenticTask]) -> int:
    payloads = [
        json.dumps(
            {"prompt": task.prompt, "setup": task.setup, "success": task.success},
            ensure_ascii=False,
            sort_keys=True,
        )
        for task in tasks
    ]
    return len(payloads) - len(set(payloads))


def validate_repeat_power_design(
    design: dict[str, object],
    tasks: list[AgenticTask],
    *,
    cells: list[LoopPolicyCell],
    model_family: str | None,
) -> None:
    """Refuse a run that does not match its predeclared sample, family, or policy contract."""
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise ValueError(f"repeat-power design schema_version must be {DESIGN_SCHEMA_VERSION}")
    planned_n = int(cast(int, design["planned_n"]))
    if len(tasks) != planned_n:
        raise ValueError(f"repeat-power design declares {planned_n} tasks, got {len(tasks)}")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("repeat-power tasks must have unique ids")
    if _duplicate_payloads(tasks):
        raise ValueError("repeat-power tasks must have non-duplicate prompt/setup/success payloads")

    required = cast(dict[str, int], design["required_task_families"])
    counts = _family_counts(tasks)
    missing = {
        family: minimum for family, minimum in required.items() if counts[family] < int(minimum)
    }
    if missing:
        raise ValueError(f"repeat-power task-family coverage is short: {missing}")
    if counts[""]:
        raise ValueError("every repeat-power task must carry a family label")

    mde = float(cast(float, design["minimum_detectable_completion_gain"]))
    minimum_discordant = int(cast(int, design["minimum_discordant_pairs"]))
    if not 0.0 < mde <= 1.0:
        raise ValueError("minimum_detectable_completion_gain must be in (0, 1]")
    if planned_n * mde < minimum_discordant:
        raise ValueError(
            "planned task count cannot make the declared completion gain reach "
            "the discordant-pair gate"
        )

    expected_cells = {
        (BASELINE_MAX_STEPS, MALFORMED_ANSWER, REPEATED_ALLOW),
        (BASELINE_MAX_STEPS, MALFORMED_ANSWER, REPEATED_NOOP),
    }
    actual_cells = {
        (cell.max_steps, cell.policy.malformed_call, cell.policy.repeated_call) for cell in cells
    }
    if actual_cells != expected_cells:
        raise ValueError(
            "repeat-power design requires exactly steps=6, malformed=answer, repeat=allow,noop"
        )
    families = cast(list[str], design["required_model_families"])
    if model_family is None or model_family not in families:
        raise ValueError(f"model_family must be one of the predeclared families: {families}")


def _activation(report: LoopPolicyReport, tasks: list[AgenticTask]) -> dict[str, object]:
    active = [episode.n_repeated_calls > 0 for episode in report.episodes]
    by_family: dict[str, dict[str, object]] = {}
    for family in sorted({task.family or "" for task in tasks}):
        indexes = [index for index, task in enumerate(tasks) if task.family == family]
        count = sum(active[index] for index in indexes)
        by_family[family] = {
            "activated_tasks": count,
            "tasks": len(indexes),
            "activation_rate": count / len(indexes),
        }
    return {
        "activated_tasks": sum(active),
        "tasks": len(active),
        "activation_rate": sum(active) / len(active),
        "by_family": by_family,
    }


def _cost_gate(
    report: LoopPolicyReport,
    baseline: LoopPolicyReport,
    metric: str,
    relative_limit: float,
) -> dict[str, object]:
    comparison = report.paired[metric]
    baseline_mean = baseline.metric_mean(metric)
    allowed_delta = baseline_mean * relative_limit
    observed = comparison["delta"]
    return {
        "metric": metric,
        "relative_increase_limit": relative_limit,
        "baseline_mean": baseline_mean,
        "allowed_delta": allowed_delta,
        "paired_delta": observed,
        "passed": observed["hi"] <= allowed_delta,
    }


def analyze_repeat_power(
    design: dict[str, object],
    tasks: list[AgenticTask],
    reports: list[LoopPolicyReport],
    *,
    model_family: str | None,
) -> dict[str, object]:
    """Resolve coverage, activation, completion, and paired cost gates for one model family."""
    baseline = next(report for report in reports if report.cell.is_baseline)
    noop = next(report for report in reports if report.cell.policy.repeated_call == REPEATED_NOOP)
    baseline_activation = _activation(baseline, tasks)
    noop_activation = _activation(noop, tasks)
    minimum_activation_rate = float(cast(float, design["minimum_activation_rate"]))
    minimum_family_activation = int(cast(int, design["minimum_activated_tasks_per_family"]))
    family_activation_passed = all(
        cast(int, row["activated_tasks"]) >= minimum_family_activation
        for row in cast(dict[str, dict[str, object]], baseline_activation["by_family"]).values()
    )
    activation_passed = (
        cast(float, baseline_activation["activation_rate"]) >= minimum_activation_rate
        and family_activation_passed
    )

    completion = noop.paired[METRIC_COMPLETION]
    mde = float(cast(float, design["minimum_detectable_completion_gain"]))
    completion_passed = completion["delta"]["mean"] >= mde and reading_of(completion) == "separated"
    cost_limits = cast(dict[str, float], design["maximum_relative_cost_increase"])
    prompt_gate = _cost_gate(
        noop,
        baseline,
        METRIC_PROMPT_TOKENS,
        float(cost_limits["total_model_input_tokens"]),
    )
    wall_gate = _cost_gate(
        noop,
        baseline,
        METRIC_WALL_CLOCK,
        float(cost_limits["elapsed_s"]),
    )
    supports_noop = bool(
        activation_passed and completion_passed and prompt_gate["passed"] and wall_gate["passed"]
    )
    return {
        "study_id": design["study_id"],
        "model_family": model_family,
        "task_family_counts": dict(sorted(_family_counts(tasks).items())),
        "coverage_passed": True,
        "activation": {
            REPEATED_ALLOW: baseline_activation,
            REPEATED_NOOP: noop_activation,
            "minimum_rate": minimum_activation_rate,
            "minimum_tasks_per_family": minimum_family_activation,
            "passed": activation_passed,
        },
        "completion": {
            "minimum_detectable_gain": mde,
            "paired_delta": completion["delta"],
            "reading": reading_of(completion),
            "wins": completion["wins"],
            "losses": completion["losses"],
            "ties": completion["ties"],
            "passed": completion_passed,
        },
        "cost": {
            METRIC_PROMPT_TOKENS: prompt_gate,
            METRIC_WALL_CLOCK: wall_gate,
            "passed": bool(prompt_gate["passed"] and wall_gate["passed"]),
        },
        "supports_noop": supports_noop,
        "reason": (
            "coverage, activation, material completion, and paired cost gates all pass"
            if supports_noop
            else "one or more prospective repeat-power gates did not pass"
        ),
    }


__all__ = [
    "DESIGN_SCHEMA_VERSION",
    "analyze_repeat_power",
    "load_repeat_power_design",
    "validate_repeat_power_design",
]
