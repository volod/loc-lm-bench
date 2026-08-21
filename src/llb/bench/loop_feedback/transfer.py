"""Reading a finished neutral repeat-feedback transfer: did the notice carry across families?

The prospective contract these runs were produced under is validated in
`agentic_loop_feedback_transfer_design`; this module only reads the seeded runs against it.
"""

from dataclasses import dataclass
from typing import cast

from llb.bench.agentic.loop_policy import DEFAULT_REPEAT_FEEDBACK
from llb.bench.agentic.design_fields import as_float, as_int, as_ints, as_mapping
from llb.bench.loop_feedback.outcomes import (
    compact_family_outcomes,
    paired_delta,
)
from llb.bench.loop_feedback.transfer_design import MODEL_FAMILY, candidate_variant_of
from llb.bench.loop_policy.report import METRIC_PROMPT_TOKENS, METRIC_WALL_CLOCK


@dataclass(frozen=True, slots=True)
class FeedbackTransferRun:
    """One seeded Gemma comparison plus its persisted cell manifests."""

    seed: int
    model: str
    analysis: dict[str, object]
    manifests: dict[str, str]


def _check_run_isolation(run: FeedbackTransferRun, candidate: str) -> None:
    """One run is the seed it claims, on the declared family, carrying only its own candidate."""
    if (
        run.analysis.get("model_family") != MODEL_FAMILY
        or run.analysis.get("run_seed") != run.seed
        or set(as_mapping(run.analysis, "variants")) != {candidate}
    ):
        raise ValueError("transfer run metadata or candidate isolation is invalid")


def _family_rates(redirect: dict[str, object]) -> dict[str, float]:
    """Each task family's response rate under one arm, in family order."""
    by_family = cast(dict[str, dict[str, object]], redirect["by_family"])
    return {family: as_float(row, "response_rate") for family, row in sorted(by_family.items())}


def _transfer_seed_row(
    run: FeedbackTransferRun, candidate: str, *, response_floor: float, minimum_families: int
) -> dict[str, object]:
    """One seed: which families answered the notice, and every gate the arm has to clear."""
    baseline = as_mapping(run.analysis, "baseline")
    variant = cast(dict[str, dict[str, object]], run.analysis["variants"])[candidate]
    redirect = as_mapping(variant, "redirect")
    baseline_redirect = as_mapping(baseline, "redirect")
    rates = _family_rates(redirect)
    baseline_rates = _family_rates(baseline_redirect)
    responsive = [family for family, rate in rates.items() if rate >= response_floor]
    family_passed = len(responsive) >= minimum_families
    completion = as_mapping(variant, "completion")
    cost = as_mapping(variant, "cost")
    prompt_cost = as_mapping(cost, METRIC_PROMPT_TOKENS)
    wall_cost = as_mapping(cost, METRIC_WALL_CLOCK)
    eligible = bool(
        run.analysis["coverage_passed"]
        and baseline["activation_passed"]
        and variant["activation_passed"]
    )
    return {
        "seed": run.seed,
        "model": run.model,
        "eligible": eligible,
        "coverage_passed": run.analysis["coverage_passed"],
        "baseline_activation_passed": baseline["activation_passed"],
        "candidate_activation_passed": variant["activation_passed"],
        "baseline_task_family_response_rates": baseline_rates,
        "task_family_response_rates": rates,
        "task_family_response_completion": compact_family_outcomes(
            cast(dict[str, dict[str, object]], redirect["by_family"])
        ),
        "task_family_response_rate_deltas": {
            family: rate - baseline_rates[family] for family, rate in rates.items()
        },
        "responsive_task_families": responsive,
        "task_family_response_gate_passed": family_passed,
        "baseline_response_rate": baseline_redirect["response_rate"],
        "response_rate": redirect["response_rate"],
        "response_rate_delta": as_float(redirect, "response_rate")
        - as_float(baseline_redirect, "response_rate"),
        "completion_rate": variant["completion_rate"],
        "completion_delta": paired_delta(variant, "completion"),
        "completion_comparison": completion["paired"],
        "completion_gate_passed": completion["passed"],
        "prompt_token_delta": paired_delta(variant, METRIC_PROMPT_TOKENS),
        "prompt_cost_gate": prompt_cost,
        "prompt_cost_gate_passed": prompt_cost["passed"],
        "wall_clock_delta_s": paired_delta(variant, METRIC_WALL_CLOCK),
        "wall_cost_gate": wall_cost,
        "wall_cost_gate_passed": wall_cost["passed"],
        "supports_transfer": bool(eligible and family_passed and variant["supports_variant"]),
        "manifests": run.manifests,
    }


def analyze_feedback_transfer(
    design: dict[str, object], runs: list[FeedbackTransferRun]
) -> dict[str, object]:
    """Require useful redirects across task families on both immutable seeds."""
    seeds = as_ints(design, "run_seeds")
    if len(runs) != len(seeds) or {run.seed for run in runs} != set(seeds):
        raise ValueError("transfer runs do not match the exact predeclared seed grid")
    candidate = candidate_variant_of(design)
    response_rule = as_mapping(design, "task_family_response_rule")
    response_floor = as_float(response_rule, "minimum_response_rate")
    minimum_families = as_int(response_rule, "minimum_supported_task_families_per_seed")
    seed_rows: list[dict[str, object]] = []
    for run in sorted(runs, key=lambda item: item.seed):
        _check_run_isolation(run, candidate)
        seed_rows.append(
            _transfer_seed_row(
                run,
                candidate,
                response_floor=response_floor,
                minimum_families=minimum_families,
            )
        )
    supported_seeds = sum(bool(row["supports_transfer"]) for row in seed_rows)
    required_seeds = as_int(response_rule, "minimum_supported_seeds")
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
