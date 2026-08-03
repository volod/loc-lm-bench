"""Prospective Gemma repeat-feedback transfer across fresh task families."""

import re
from dataclasses import dataclass
from typing import cast

from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    REPEATED_NOOP_OBSERVATIONS,
    REPEAT_FEEDBACK_GEMMA_PROGRESS,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_context import task_set_digest
from llb.bench.agentic_loop_feedback import validate_repeat_feedback_design
from llb.bench.agentic_loop_feedback_outcomes import compact_family_outcomes
from llb.bench.agentic_loop_policy import policy_grid
from llb.bench.agentic_loop_policy_report import METRIC_PROMPT_TOKENS, METRIC_WALL_CLOCK

STUDY_KIND = "repeat_feedback_task_family_transfer"
MODEL_FAMILY = "gemma"
EXPECTED_HYPOTHESIS = (
    "Completed-state feedback will make Gemma advance after a suppressed repeat in at least "
    "three task families while preserving material completion and paired cost bounds."
)
FORBIDDEN_NOTICE_TERMS = (
    "answer",
    "calculator",
    "database",
    "file",
    "mutation",
    "read",
    "search",
    "tool",
    "write",
)


@dataclass(frozen=True, slots=True)
class FeedbackTransferRun:
    """One seeded Gemma comparison plus its persisted cell manifests."""

    seed: int
    model: str
    analysis: dict[str, object]
    manifests: dict[str, str]


def _candidate(design: dict[str, object]) -> str:
    return cast(str, design["candidate_feedback_variant"])


def validate_feedback_transfer_design(design: dict[str, object], tasks: list[AgenticTask]) -> None:
    """Refuse inference unless the neutral notice, fresh ledger, seeds, and gates are fixed."""
    validate_neutral_feedback_design(
        design,
        tasks,
        study_kind=STUDY_KIND,
        hypothesis=EXPECTED_HYPOTHESIS,
        candidate_variant=REPEAT_FEEDBACK_GEMMA_PROGRESS,
    )


def validate_neutral_feedback_design(
    design: dict[str, object],
    tasks: list[AgenticTask],
    *,
    study_kind: str,
    hypothesis: str,
    candidate_variant: str,
) -> None:
    """Validate one prospective neutral-feedback contract before model inference."""
    if design.get("study_kind") != study_kind:
        raise ValueError(f"study_kind must be {study_kind}")
    if design.get("hypothesis") != hypothesis:
        raise ValueError("transfer hypothesis does not match the immutable prospective hypothesis")
    reference = cast(dict[str, object], design["reference"])
    digest = task_set_digest(tasks)
    if reference.get("task_set_digest") != digest:
        raise ValueError("transfer task digest does not match the predeclared holdout ledger")
    excluded = cast(list[str], reference["excluded_prior_task_set_digests"])
    if not excluded or digest in excluded:
        raise ValueError("transfer ledger must be fresh relative to every excluded prior digest")

    families = cast(list[str], design["required_model_families"])
    roster = cast(list[dict[str, object]], design["roster"])
    if families != [MODEL_FAMILY] or len(roster) != 1:
        raise ValueError("task-family transfer requires exactly one Gemma roster row")
    if (
        roster[0].get("model_family") != MODEL_FAMILY
        or roster[0].get("backend") != "ollama"
        or not roster[0].get("model")
    ):
        raise ValueError("transfer roster must declare one installed Ollama Gemma model")

    candidate = _candidate(design)
    if candidate != candidate_variant:
        raise ValueError("transfer candidate must be the immutable registered notice")
    variants = cast(list[str], design["repeat_feedback_variants"])
    if variants != [DEFAULT_REPEAT_FEEDBACK, candidate]:
        raise ValueError("transfer must isolate current versus its registered candidate feedback")
    notice = cast(str, design["notice_text"])
    if notice != REPEATED_NOOP_OBSERVATIONS[candidate]:
        raise ValueError("transfer notice does not match the registered immutable text")
    if not notice.isascii() or len(notice) > int(cast(int, design["maximum_notice_chars"])):
        raise ValueError("transfer notice violates the concise ASCII controller contract")
    forbidden = cast(list[str], design["forbidden_notice_terms"])
    notice_words = set(re.findall(r"[a-z]+", notice.casefold()))
    if forbidden != list(FORBIDDEN_NOTICE_TERMS) or notice_words.intersection(forbidden):
        raise ValueError("transfer notice contains task-specific or answer-specific language")

    seeds = cast(list[int], design["run_seeds"])
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("task-family transfer requires exactly two unique seeds")
    sampling = cast(dict[str, object], design["sampling"])
    if not 0.0 < float(cast(float, sampling["temperature"])) <= 1.0:
        raise ValueError("seeded repetitions require temperature in (0, 1]")
    response_rule = cast(dict[str, object], design["task_family_response_rule"])
    response_floor = float(cast(float, response_rule["minimum_response_rate"]))
    minimum_families = int(cast(int, response_rule["minimum_supported_task_families_per_seed"]))
    required_families = cast(dict[str, int], design["required_task_families"])
    if not 0.0 < response_floor <= 1.0 or not 3 <= minimum_families <= len(required_families):
        raise ValueError("task-family response thresholds are outside the declared ledger")
    if int(cast(int, response_rule["minimum_supported_seeds"])) != len(seeds):
        raise ValueError("transfer adoption must require every predeclared seed")

    fixed = cast(dict[str, object], design["fixed_policy"])
    validate_repeat_feedback_design(
        design,
        tasks,
        cells=policy_grid(
            [int(cast(int, fixed["max_steps"]))],
            [cast(str, fixed["malformed_call"])],
            cast(list[str], fixed["repeated_call"]),
            variants,
        ),
        model_family=MODEL_FAMILY,
        run_seed=seeds[0],
    )


def _delta(row: dict[str, object], metric: str) -> float:
    if metric == "completion":
        paired = cast(dict[str, object], cast(dict[str, object], row["completion"])["paired"])
        return float(cast(dict[str, float], paired["delta"])["mean"])
    gate = cast(dict[str, object], cast(dict[str, object], row["cost"])[metric])
    return float(cast(dict[str, float], gate["paired_delta"])["mean"])


def analyze_feedback_transfer(
    design: dict[str, object], runs: list[FeedbackTransferRun]
) -> dict[str, object]:
    """Require useful redirects across task families on both immutable seeds."""
    seeds = cast(list[int], design["run_seeds"])
    if len(runs) != len(seeds) or {run.seed for run in runs} != set(seeds):
        raise ValueError("transfer runs do not match the exact predeclared seed grid")
    candidate = _candidate(design)
    response_rule = cast(dict[str, object], design["task_family_response_rule"])
    response_floor = float(cast(float, response_rule["minimum_response_rate"]))
    minimum_families = int(cast(int, response_rule["minimum_supported_task_families_per_seed"]))
    seed_rows: list[dict[str, object]] = []
    for run in sorted(runs, key=lambda item: item.seed):
        if (
            run.analysis.get("model_family") != MODEL_FAMILY
            or run.analysis.get("run_seed") != run.seed
            or set(cast(dict[str, object], run.analysis["variants"])) != {candidate}
        ):
            raise ValueError("transfer run metadata or candidate isolation is invalid")
        baseline = cast(dict[str, object], run.analysis["baseline"])
        variant = cast(dict[str, dict[str, object]], run.analysis["variants"])[candidate]
        redirect = cast(dict[str, object], variant["redirect"])
        baseline_redirect = cast(dict[str, object], baseline["redirect"])
        by_family = cast(dict[str, dict[str, object]], redirect["by_family"])
        baseline_by_family = cast(dict[str, dict[str, object]], baseline_redirect["by_family"])
        family_rates = {
            family: float(cast(float, row["response_rate"]))
            for family, row in sorted(by_family.items())
        }
        baseline_family_rates = {
            family: float(cast(float, row["response_rate"]))
            for family, row in sorted(baseline_by_family.items())
        }
        family_outcomes = compact_family_outcomes(by_family)
        responsive_families = [
            family for family, rate in family_rates.items() if rate >= response_floor
        ]
        family_response_passed = len(responsive_families) >= minimum_families
        completion = cast(dict[str, object], variant["completion"])
        cost = cast(dict[str, object], variant["cost"])
        prompt_cost = cast(dict[str, object], cost[METRIC_PROMPT_TOKENS])
        wall_cost = cast(dict[str, object], cost[METRIC_WALL_CLOCK])
        eligible = bool(
            run.analysis["coverage_passed"]
            and baseline["activation_passed"]
            and variant["activation_passed"]
        )
        supports = bool(eligible and family_response_passed and variant["supports_variant"])
        seed_rows.append(
            {
                "seed": run.seed,
                "model": run.model,
                "eligible": eligible,
                "coverage_passed": run.analysis["coverage_passed"],
                "baseline_activation_passed": baseline["activation_passed"],
                "candidate_activation_passed": variant["activation_passed"],
                "baseline_task_family_response_rates": baseline_family_rates,
                "task_family_response_rates": family_rates,
                "task_family_response_completion": family_outcomes,
                "task_family_response_rate_deltas": {
                    family: rate - baseline_family_rates[family]
                    for family, rate in family_rates.items()
                },
                "responsive_task_families": responsive_families,
                "task_family_response_gate_passed": family_response_passed,
                "baseline_response_rate": baseline_redirect["response_rate"],
                "response_rate": redirect["response_rate"],
                "response_rate_delta": float(cast(float, redirect["response_rate"]))
                - float(cast(float, baseline_redirect["response_rate"])),
                "completion_rate": variant["completion_rate"],
                "completion_delta": _delta(variant, "completion"),
                "completion_comparison": completion["paired"],
                "completion_gate_passed": completion["passed"],
                "prompt_token_delta": _delta(variant, METRIC_PROMPT_TOKENS),
                "prompt_cost_gate": prompt_cost,
                "prompt_cost_gate_passed": prompt_cost["passed"],
                "wall_clock_delta_s": _delta(variant, METRIC_WALL_CLOCK),
                "wall_cost_gate": wall_cost,
                "wall_cost_gate_passed": wall_cost["passed"],
                "supports_transfer": supports,
                "manifests": run.manifests,
            }
        )
    supported_seeds = sum(bool(row["supports_transfer"]) for row in seed_rows)
    required_seeds = int(cast(int, response_rule["minimum_supported_seeds"]))
    supports_transfer = supported_seeds >= required_seeds
    return {
        "study_id": design["study_id"],
        "model_family": MODEL_FAMILY,
        "candidate_feedback_variant": candidate,
        "candidate_isolation_passed": True,
        "run_seeds": seeds,
        "sampling": design["sampling"],
        "task_family_response_rule": response_rule,
        "seed_rows": seed_rows,
        "supported_seeds": supported_seeds,
        "required_supported_seeds": required_seeds,
        "supports_task_family_transfer": supports_transfer,
        "recommended_feedback_variant": candidate if supports_transfer else DEFAULT_REPEAT_FEEDBACK,
        "reason": (
            "the neutral notice clears task-family response, completion, and cost gates on both seeds"
            if supports_transfer
            else "the neutral notice does not clear every predeclared transfer gate on both seeds"
        ),
    }
