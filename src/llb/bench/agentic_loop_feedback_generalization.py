"""Cross-family, seeded stability analysis for localized repeat feedback."""

from dataclasses import dataclass
from typing import cast

from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    REPEAT_FEEDBACK_BILINGUAL,
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

STUDY_KIND = "repeat_feedback_generalization"


@dataclass(frozen=True, slots=True)
class FeedbackSeedRun:
    """One model-family/seed decision plus its persisted cell manifests."""

    model_family: str
    model: str
    seed: int
    analysis: dict[str, object]
    manifests: dict[str, str]


def _roster(design: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], design["roster"])


def _check_generalization_identity(design: dict[str, object], tasks: list[AgenticTask]) -> None:
    """The study, its task ledger, and the two arms it isolates."""
    if design.get("study_kind") != STUDY_KIND:
        raise ValueError(f"study_kind must be {STUDY_KIND}")
    reference = as_mapping(design, "reference")
    if reference.get("task_set_digest") != task_set_digest(tasks):
        raise ValueError("generalization task digest does not match the predeclared ledger")
    if as_strs(design, "repeat_feedback_variants") != [
        DEFAULT_REPEAT_FEEDBACK,
        REPEAT_FEEDBACK_BILINGUAL,
    ]:
        raise ValueError("generalization must isolate current versus bilingual feedback")


def _validated_families(design: dict[str, object]) -> list[str]:
    """Four or more unique families, each on the roster once, in declared order."""
    families = as_strs(design, "required_model_families")
    roster = _roster(design)
    roster_families = [as_str(row, "model_family") for row in roster]
    if len(families) < 4 or len(set(families)) != len(families):
        raise ValueError("generalization needs at least four unique model families")
    if roster_families != families or len(set(roster_families)) != len(roster_families):
        raise ValueError("roster must contain each required model family exactly once and in order")
    if any(row.get("backend") != "ollama" or not row.get("model") for row in roster):
        raise ValueError("every generalization roster row needs an Ollama model")
    return families


def _check_family_reach(design: dict[str, object], families: list[str]) -> None:
    """How far past the families that already answered this question the roster has to reach."""
    reference_families = as_strs(design, "reference_model_families")
    if not set(reference_families).issubset(families):
        raise ValueError("reference model families must be present in the roster")
    additional = len(set(families) - set(reference_families))
    if additional < as_int(design, "minimum_additional_model_families"):
        raise ValueError("generalization roster is short of additional model families")


def _validated_seeds(design: dict[str, object]) -> list[int]:
    """At least two unique seeds at a decodable temperature."""
    seeds = as_ints(design, "run_seeds")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("run_seeds must contain at least two unique values")
    if not 0.0 < as_float(as_mapping(design, "sampling"), "temperature") <= 1.0:
        raise ValueError("seeded repetitions require temperature in (0, 1]")
    return seeds


def _check_adoption_rule(design: dict[str, object], families: list[str], seeds: list[int]) -> None:
    """What breadth of family support an adoption needs, and on how many seeds."""
    rule = as_mapping(design, "cross_family_adoption_rule")
    minimum_families = as_int(rule, "minimum_supported_families")
    minimum_fraction = as_float(rule, "minimum_supported_fraction")
    if not 1 <= minimum_families <= len(families) or not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("cross-family support thresholds are outside the roster range")
    if as_int(rule, "minimum_supported_seeds_per_family") != len(seeds):
        raise ValueError("family routing must require support on every predeclared seed")
    if not isinstance(rule.get("require_additional_family_support"), bool):
        raise ValueError("require_additional_family_support must be boolean")


def validate_feedback_generalization_design(
    design: dict[str, object], tasks: list[AgenticTask]
) -> None:
    """Refuse a generalization contract with mutable or dependent study dimensions."""
    _check_generalization_identity(design, tasks)
    families = _validated_families(design)
    _check_family_reach(design, families)
    seeds = _validated_seeds(design)
    _check_adoption_rule(design, families, seeds)
    fixed = as_mapping(design, "fixed_policy")
    validate_repeat_feedback_design(
        design,
        tasks,
        cells=policy_grid(
            [as_int(fixed, "max_steps")],
            [as_str(fixed, "malformed_call")],
            as_strs(fixed, "repeated_call"),
            [DEFAULT_REPEAT_FEEDBACK, REPEAT_FEEDBACK_BILINGUAL],
        ),
        model_family=families[0],
        run_seed=seeds[0],
    )


def _variant_row(run: FeedbackSeedRun) -> dict[str, object]:
    variants = cast(dict[str, dict[str, object]], run.analysis["variants"])
    return variants[REPEAT_FEEDBACK_BILINGUAL]


def _delta(row: dict[str, object], metric: str) -> float:
    if metric == "completion":
        paired = cast(dict[str, object], cast(dict[str, object], row["completion"])["paired"])
        return float(cast(dict[str, float], paired["delta"])["mean"])
    gate = cast(dict[str, object], cast(dict[str, object], row["cost"])[metric])
    return float(cast(dict[str, float], gate["paired_delta"])["mean"])


def _check_run_grid(runs: list[FeedbackSeedRun], families: list[str], seeds: list[int]) -> None:
    """Every family/seed cell ran exactly once, and each run is the coordinate it claims."""
    expected = {(family, seed) for family in families for seed in seeds}
    actual = {(run.model_family, run.seed) for run in runs}
    if len(runs) != len(actual) or actual != expected:
        raise ValueError("generalization runs do not match the exact family/seed grid")
    if any(
        run.analysis.get("model_family") != run.model_family
        or run.analysis.get("run_seed") != run.seed
        for run in runs
    ):
        raise ValueError("run analysis metadata does not match its family/seed coordinate")


def _seed_row(run: FeedbackSeedRun) -> dict[str, object]:
    """One family/seed cell: what it measured, whether it may be read, and what it supports."""
    baseline = as_mapping(run.analysis, "baseline")
    variant = _variant_row(run)
    coverage_passed = bool(run.analysis["coverage_passed"])
    baseline_activation_passed = bool(baseline["activation_passed"])
    variant_activation_passed = bool(variant["activation_passed"])
    eligible = bool(coverage_passed and baseline_activation_passed and variant_activation_passed)
    return {
        "model_family": run.model_family,
        "model": run.model,
        "seed": run.seed,
        "coverage_passed": coverage_passed,
        "baseline_activation_rate": baseline["activation_rate"],
        "baseline_activation_passed": baseline_activation_passed,
        "variant_activation_rate": variant["activation_rate"],
        "variant_activation_passed": variant_activation_passed,
        "eligible": eligible,
        "response_rate": as_float(as_mapping(variant, "redirect"), "response_rate"),
        "completion_rate": variant["completion_rate"],
        "completion_delta": _delta(variant, "completion"),
        "prompt_token_delta": _delta(variant, METRIC_PROMPT_TOKENS),
        "wall_clock_delta_s": _delta(variant, METRIC_WALL_CLOCK),
        "supports_bilingual": bool(eligible and variant["supports_variant"]),
        "manifests": run.manifests,
    }


def _family_row(rows: list[dict[str, object]], stable_seed_count: int) -> dict[str, object]:
    """One family's verdict across its seeds: how many supported, and what it routes to."""
    support_count = sum(1 for row in rows if row["supports_bilingual"])
    stable = support_count >= stable_seed_count
    deltas = [as_float(row, "completion_delta") for row in rows]
    rates = [as_float(row, "response_rate") for row in rows]
    return {
        "model": rows[0]["model"],
        "seeds": [row["seed"] for row in rows],
        "supported_seeds": support_count,
        "stable_support": stable,
        "routed_feedback_variant": (
            REPEAT_FEEDBACK_BILINGUAL if stable else DEFAULT_REPEAT_FEEDBACK
        ),
        "completion_delta_range": [min(deltas), max(deltas)],
        "response_rate_range": [min(rates), max(rates)],
    }


def _supports_global_default(
    rule: dict[str, object],
    families: list[str],
    stable_families: list[str],
    additional_supported: list[str],
    all_cells_eligible: bool,
) -> bool:
    """The predeclared cross-family rule, applied to what the families actually supported."""
    return bool(
        all_cells_eligible
        and len(stable_families) >= as_int(rule, "minimum_supported_families")
        and len(stable_families) / len(families) >= as_float(rule, "minimum_supported_fraction")
        and (additional_supported or not rule["require_additional_family_support"])
    )


def analyze_feedback_generalization(
    design: dict[str, object], runs: list[FeedbackSeedRun]
) -> dict[str, object]:
    """Aggregate exact family/seed decisions under the predeclared stability rule."""
    families = as_strs(design, "required_model_families")
    seeds = as_ints(design, "run_seeds")
    _check_run_grid(runs, families, seeds)
    rule = as_mapping(design, "cross_family_adoption_rule")
    stable_seed_count = as_int(rule, "minimum_supported_seeds_per_family")
    seed_rows: list[dict[str, object]] = []
    family_rows: dict[str, dict[str, object]] = {}
    for family in families:
        family_runs = sorted(
            (run for run in runs if run.model_family == family), key=lambda run: run.seed
        )
        rows = [_seed_row(run) for run in family_runs]
        seed_rows.extend(rows)
        family_rows[family] = _family_row(rows, stable_seed_count)

    all_cells_eligible = all(row["eligible"] for row in seed_rows)
    stable_families = [name for name, row in family_rows.items() if row["stable_support"]]
    reference_families = set(as_strs(design, "reference_model_families"))
    additional_supported = [name for name in stable_families if name not in reference_families]
    global_support = _supports_global_default(
        rule, families, stable_families, additional_supported, all_cells_eligible
    )
    return {
        "study_id": design["study_id"],
        "coverage_and_activation_passed": all_cells_eligible,
        "run_seeds": seeds,
        "sampling": design["sampling"],
        "seed_rows": seed_rows,
        "families": family_rows,
        "stable_supported_families": stable_families,
        "additional_supported_families": additional_supported,
        "supported_family_fraction": len(stable_families) / len(families),
        "cross_family_adoption_rule": rule,
        "supports_global_feedback_default": global_support,
        "recommended_global_feedback_variant": (
            REPEAT_FEEDBACK_BILINGUAL if global_support else DEFAULT_REPEAT_FEEDBACK
        ),
        "reason": (
            "bilingual feedback clears the predeclared cross-family and replicate-stability rule"
            if global_support
            else "bilingual feedback does not clear the predeclared cross-family stability rule"
        ),
    }
