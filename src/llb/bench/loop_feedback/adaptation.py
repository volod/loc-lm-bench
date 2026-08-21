"""Reading a finished family-specific repeat-feedback study: which notice routes per family.

The prospective contract these runs were produced under is validated in
`agentic_loop_feedback_adaptation_design`; this module only reads the seeded runs against it.
"""

from dataclasses import dataclass
from typing import cast

from llb.bench.agentic.loop_policy import DEFAULT_REPEAT_FEEDBACK
from llb.bench.agentic.design_fields import as_float, as_int, as_ints, as_mapping, as_str
from llb.bench.loop_feedback.adaptation_design import candidate_of, roster_rows
from llb.bench.loop_feedback.outcomes import paired_delta
from llb.bench.loop_policy.report import METRIC_PROMPT_TOKENS, METRIC_WALL_CLOCK


@dataclass(frozen=True, slots=True)
class FeedbackAdaptationRun:
    """One family/seed candidate comparison and its persisted cell manifests."""

    model_family: str
    model: str
    seed: int
    candidate_variant: str
    analysis: dict[str, object]
    manifests: dict[str, str]


def _check_adaptation_grid(
    design: dict[str, object], runs: list[FeedbackAdaptationRun], seeds: list[int]
) -> None:
    """Every family/seed cell ran once, against the candidate its own family declared."""
    expected = {
        (as_str(row, "model_family"), seed): candidate_of(row)
        for row in roster_rows(design)
        for seed in seeds
    }
    actual = {(run.model_family, run.seed): run.candidate_variant for run in runs}
    if len(runs) != len(actual) or actual != expected:
        raise ValueError("adaptation runs do not match the exact family/seed/candidate grid")


def _check_candidate_isolation(run: FeedbackAdaptationRun, family: str, candidate: str) -> None:
    """One run carries its own coordinate and its own family's candidate, and nothing else."""
    if (
        run.analysis.get("model_family") != family
        or run.analysis.get("run_seed") != run.seed
        or set(as_mapping(run.analysis, "variants")) != {candidate}
    ):
        raise ValueError("run analysis metadata or candidate isolation is invalid")


def _adaptation_seed_row(
    run: FeedbackAdaptationRun, family: str, candidate: str
) -> dict[str, object]:
    """One family/seed cell: every gate it passed, every delta it moved, what it supports."""
    baseline = as_mapping(run.analysis, "baseline")
    variant = cast(dict[str, dict[str, object]], run.analysis["variants"])[candidate]
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
        "model_family": family,
        "model": run.model,
        "seed": run.seed,
        "candidate_feedback_variant": candidate,
        "eligible": eligible,
        "coverage_passed": run.analysis["coverage_passed"],
        "baseline_activation_passed": baseline["activation_passed"],
        "candidate_activation_passed": variant["activation_passed"],
        "completion_gate_passed": completion["passed"],
        "prompt_cost_gate_passed": prompt_cost["passed"],
        "wall_cost_gate_passed": wall_cost["passed"],
        "cost_gate_passed": cost["passed"],
        "baseline_activation_rate": baseline["activation_rate"],
        "candidate_activation_rate": variant["activation_rate"],
        "response_rate": as_mapping(variant, "redirect")["response_rate"],
        "completion_rate": variant["completion_rate"],
        "completion_delta": paired_delta(variant, "completion"),
        "prompt_token_delta": paired_delta(variant, METRIC_PROMPT_TOKENS),
        "wall_clock_delta_s": paired_delta(variant, METRIC_WALL_CLOCK),
        "supports_candidate": bool(eligible and variant["supports_variant"]),
        "manifests": run.manifests,
    }


def _adaptation_family_row(
    rows: list[dict[str, object]], candidate: str, stable_required: int
) -> dict[str, object]:
    """One family's route: how many of its seeds supported the candidate, so what it runs."""
    support_count = sum(1 for row in rows if row["supports_candidate"])
    stable = support_count >= stable_required
    return {
        "model": rows[0]["model"],
        "candidate_feedback_variant": candidate,
        "supported_seeds": support_count,
        "required_supported_seeds": stable_required,
        "stable_support": stable,
        "routed_feedback_variant": candidate if stable else DEFAULT_REPEAT_FEEDBACK,
    }


def analyze_feedback_adaptation(
    design: dict[str, object], runs: list[FeedbackAdaptationRun]
) -> dict[str, object]:
    """Resolve stable family routes without allowing a candidate to leak across families."""
    seeds = as_ints(design, "run_seeds")
    _check_adaptation_grid(design, runs, seeds)
    rule = as_mapping(design, "cross_family_adoption_rule")
    stable_required = as_int(rule, "minimum_supported_seeds_per_family")
    seed_rows: list[dict[str, object]] = []
    family_rows: dict[str, dict[str, object]] = {}
    for roster_row in roster_rows(design):
        family = as_str(roster_row, "model_family")
        candidate = candidate_of(roster_row)
        family_runs = sorted(
            (run for run in runs if run.model_family == family), key=lambda run: run.seed
        )
        for run in family_runs:
            _check_candidate_isolation(run, family, candidate)
        rows = [_adaptation_seed_row(run, family, candidate) for run in family_runs]
        seed_rows.extend(rows)
        family_rows[family] = _adaptation_family_row(rows, candidate, stable_required)

    all_eligible = all(row["eligible"] for row in seed_rows)
    stable_families = [family for family, row in family_rows.items() if row["stable_support"]]
    fraction = len(stable_families) / len(family_rows)
    cross_family = bool(
        all_eligible
        and len(stable_families) >= as_int(rule, "minimum_supported_families")
        and fraction >= as_float(rule, "minimum_supported_fraction")
    )
    return {
        "study_id": design["study_id"],
        "coverage_and_activation_passed": all_eligible,
        "candidate_isolation_passed": True,
        "run_seeds": seeds,
        "sampling": design["sampling"],
        "seed_rows": seed_rows,
        "families": family_rows,
        "stable_supported_families": stable_families,
        "supported_family_fraction": fraction,
        "cross_family_adoption_rule": rule,
        "supports_family_adapted_routing": cross_family,
        "recommended_routing_mode": "family_adapted" if cross_family else "family_isolated",
        "reason": (
            "family-specific notices clear the predeclared cross-family routing threshold"
            if cross_family
            else "family-specific notices remain isolated under the predeclared threshold"
        ),
    }
