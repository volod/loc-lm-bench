"""Prospective family-specific repeat-feedback study and seeded routing analysis."""

from dataclasses import dataclass
from typing import cast

from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    REPEATED_NOOP_OBSERVATIONS,
    REPEAT_FEEDBACK_VARIANTS,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_context import task_set_digest
from llb.bench.agentic_design_fields import (
    as_float,
    as_int,
    as_ints,
    as_mapping,
    as_str,
    as_strs,
)
from llb.bench.agentic_loop_feedback import validate_repeat_feedback_design
from llb.bench.agentic_loop_policy import policy_grid
from llb.bench.agentic_loop_policy_report import METRIC_PROMPT_TOKENS, METRIC_WALL_CLOCK

STUDY_KIND = "repeat_feedback_family_adaptation"
REQUIRED_FAMILIES = {"aya", "mistral", "gemma"}


@dataclass(frozen=True, slots=True)
class FeedbackAdaptationRun:
    """One family/seed candidate comparison and its persisted cell manifests."""

    model_family: str
    model: str
    seed: int
    candidate_variant: str
    analysis: dict[str, object]
    manifests: dict[str, str]


def _roster(design: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], design["roster"])


def _candidate(row: dict[str, object]) -> str:
    return cast(str, row["candidate_feedback_variant"])


def _check_adaptation_ledger(design: dict[str, object], tasks: list[AgenticTask]) -> list[str]:
    """The study, its task ledger, and the families it must route over."""
    if design.get("study_kind") != STUDY_KIND:
        raise ValueError(f"study_kind must be {STUDY_KIND}")
    reference = as_mapping(design, "reference")
    if reference.get("task_set_digest") != task_set_digest(tasks):
        raise ValueError("adaptation task digest does not match the predeclared ledger")
    families = as_strs(design, "required_model_families")
    if set(families) != REQUIRED_FAMILIES or len(families) != len(REQUIRED_FAMILIES):
        raise ValueError("adaptation requires exactly the Aya, Mistral, and Gemma families")
    roster = _roster(design)
    if [row.get("model_family") for row in roster] != families:
        raise ValueError("roster must contain each required family exactly once and in order")
    if any(row.get("backend") != "ollama" or not row.get("model") for row in roster):
        raise ValueError("every adaptation roster row needs an Ollama model")
    return families


def _validated_candidates(design: dict[str, object]) -> list[str]:
    """One unique registered candidate variant per family, declared against the current default."""
    candidates = [_candidate(row) for row in _roster(design)]
    if len(set(candidates)) != len(candidates) or any(
        name == DEFAULT_REPEAT_FEEDBACK or name not in REPEAT_FEEDBACK_VARIANTS
        for name in candidates
    ):
        raise ValueError("each family needs one unique registered candidate feedback variant")
    if as_strs(design, "repeat_feedback_variants") != [DEFAULT_REPEAT_FEEDBACK, *candidates]:
        raise ValueError(
            "repeat_feedback_variants must declare current then every roster candidate"
        )
    return candidates


def _check_notice_contract(design: dict[str, object], candidates: list[str]) -> None:
    """Each candidate's wording is the registered one, concise, ASCII, and hypothesized."""
    notices = cast(dict[str, str], design["notice_text"])
    hypotheses = cast(dict[str, str], design["candidate_hypotheses"])
    maximum_notice_chars = as_int(design, "maximum_notice_chars")
    if set(notices) != set(candidates) or set(hypotheses) != set(candidates):
        raise ValueError("notice text and hypotheses must cover the exact candidate set")
    for name in candidates:
        _check_one_notice(name, notices[name], hypotheses[name], maximum_notice_chars)


def _check_one_notice(name: str, notice: str, hypothesis: str, maximum_chars: int) -> None:
    """One candidate's notice: the registered text, inside the concise controller contract."""
    if notice != REPEATED_NOOP_OBSERVATIONS[name]:
        raise ValueError(f"predeclared notice text does not match registered variant {name}")
    if not notice.isascii() or len(notice) > maximum_chars or not notice.startswith("[loop]"):
        raise ValueError(f"candidate notice {name} violates the concise controller contract")
    if not hypothesis.strip():
        raise ValueError(f"candidate hypothesis {name} must be predeclared")


def _validated_seeds(design: dict[str, object]) -> list[int]:
    """Two unique seeds at a decodable temperature, so a family route is not one lucky draw."""
    seeds = as_ints(design, "run_seeds")
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("family adaptation requires exactly two unique seeds")
    if not 0.0 < as_float(as_mapping(design, "sampling"), "temperature") <= 1.0:
        raise ValueError("seeded repetitions require temperature in (0, 1]")
    return seeds


def _check_adoption_rule(design: dict[str, object], families: list[str], seeds: int) -> None:
    """How much support a route needs before it may be adopted for a family."""
    rule = as_mapping(design, "cross_family_adoption_rule")
    if as_int(rule, "minimum_supported_seeds_per_family") != seeds:
        raise ValueError("family routing must require support on both seeds")
    minimum_families = as_int(rule, "minimum_supported_families")
    minimum_fraction = as_float(rule, "minimum_supported_fraction")
    if not 1 <= minimum_families <= len(families) or not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("cross-family thresholds are outside the roster range")


def _check_per_family_cells(design: dict[str, object], tasks: list[AgenticTask], seed: int) -> None:
    """Each family's own two-arm grid, checked through the shared repeat-feedback contract."""
    fixed = as_mapping(design, "fixed_policy")
    for row in _roster(design):
        validate_repeat_feedback_design(
            design,
            tasks,
            cells=policy_grid(
                [as_int(fixed, "max_steps")],
                [as_str(fixed, "malformed_call")],
                as_strs(fixed, "repeated_call"),
                [DEFAULT_REPEAT_FEEDBACK, _candidate(row)],
            ),
            model_family=as_str(row, "model_family"),
            run_seed=seed,
        )


def validate_feedback_adaptation_design(
    design: dict[str, object], tasks: list[AgenticTask]
) -> None:
    """Refuse inference unless wording, hypotheses, routes, seeds, and gates are fixed."""
    families = _check_adaptation_ledger(design, tasks)
    candidates = _validated_candidates(design)
    _check_notice_contract(design, candidates)
    seeds = _validated_seeds(design)
    _check_adoption_rule(design, families, len(seeds))
    _check_per_family_cells(design, tasks, seeds[0])


def _delta(row: dict[str, object], metric: str) -> float:
    if metric == "completion":
        paired = cast(dict[str, object], cast(dict[str, object], row["completion"])["paired"])
        return float(cast(dict[str, float], paired["delta"])["mean"])
    gate = cast(dict[str, object], cast(dict[str, object], row["cost"])[metric])
    return float(cast(dict[str, float], gate["paired_delta"])["mean"])


def _check_adaptation_grid(
    design: dict[str, object], runs: list[FeedbackAdaptationRun], seeds: list[int]
) -> None:
    """Every family/seed cell ran once, against the candidate its own family declared."""
    expected = {
        (as_str(row, "model_family"), seed): _candidate(row)
        for row in _roster(design)
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
        "completion_delta": _delta(variant, "completion"),
        "prompt_token_delta": _delta(variant, METRIC_PROMPT_TOKENS),
        "wall_clock_delta_s": _delta(variant, METRIC_WALL_CLOCK),
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
    for roster_row in _roster(design):
        family = as_str(roster_row, "model_family")
        candidate = _candidate(roster_row)
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
