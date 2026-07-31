"""Cross-family, seeded stability analysis for localized repeat feedback."""

from dataclasses import dataclass
from typing import cast

from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    REPEAT_FEEDBACK_BILINGUAL,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_context import task_set_digest
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


def validate_feedback_generalization_design(
    design: dict[str, object], tasks: list[AgenticTask]
) -> None:
    """Refuse a generalization contract with mutable or dependent study dimensions."""
    if design.get("study_kind") != STUDY_KIND:
        raise ValueError(f"study_kind must be {STUDY_KIND}")
    reference = cast(dict[str, object], design["reference"])
    if reference.get("task_set_digest") != task_set_digest(tasks):
        raise ValueError("generalization task digest does not match the predeclared ledger")
    variants = cast(list[str], design["repeat_feedback_variants"])
    if variants != [DEFAULT_REPEAT_FEEDBACK, REPEAT_FEEDBACK_BILINGUAL]:
        raise ValueError("generalization must isolate current versus bilingual feedback")

    families = cast(list[str], design["required_model_families"])
    roster = _roster(design)
    roster_families = [cast(str, row.get("model_family")) for row in roster]
    if len(families) < 4 or len(set(families)) != len(families):
        raise ValueError("generalization needs at least four unique model families")
    if roster_families != families or len(set(roster_families)) != len(roster_families):
        raise ValueError("roster must contain each required model family exactly once and in order")
    if any(row.get("backend") != "ollama" or not row.get("model") for row in roster):
        raise ValueError("every generalization roster row needs an Ollama model")
    reference_families = cast(list[str], design["reference_model_families"])
    minimum_additional = int(cast(int, design["minimum_additional_model_families"]))
    if not set(reference_families).issubset(families):
        raise ValueError("reference model families must be present in the roster")
    if len(set(families) - set(reference_families)) < minimum_additional:
        raise ValueError("generalization roster is short of additional model families")

    seeds = cast(list[int], design["run_seeds"])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("run_seeds must contain at least two unique values")
    sampling = cast(dict[str, object], design["sampling"])
    temperature = float(cast(float, sampling["temperature"]))
    if not 0.0 < temperature <= 1.0:
        raise ValueError("seeded repetitions require temperature in (0, 1]")

    rule = cast(dict[str, object], design["cross_family_adoption_rule"])
    minimum_families = int(cast(int, rule["minimum_supported_families"]))
    minimum_fraction = float(cast(float, rule["minimum_supported_fraction"]))
    stable_seeds = int(cast(int, rule["minimum_supported_seeds_per_family"]))
    if not 1 <= minimum_families <= len(families) or not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("cross-family support thresholds are outside the roster range")
    if stable_seeds != len(seeds):
        raise ValueError("family routing must require support on every predeclared seed")
    if not isinstance(rule.get("require_additional_family_support"), bool):
        raise ValueError("require_additional_family_support must be boolean")
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


def analyze_feedback_generalization(
    design: dict[str, object], runs: list[FeedbackSeedRun]
) -> dict[str, object]:
    """Aggregate exact family/seed decisions under the predeclared stability rule."""
    families = cast(list[str], design["required_model_families"])
    seeds = cast(list[int], design["run_seeds"])
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

    rule = cast(dict[str, object], design["cross_family_adoption_rule"])
    stable_seed_count = int(cast(int, rule["minimum_supported_seeds_per_family"]))
    seed_rows: list[dict[str, object]] = []
    family_rows: dict[str, dict[str, object]] = {}
    all_cells_eligible = True
    for family in families:
        family_runs = sorted(
            (run for run in runs if run.model_family == family), key=lambda run: run.seed
        )
        support_count = 0
        completion_deltas: list[float] = []
        response_rates: list[float] = []
        for run in family_runs:
            baseline = cast(dict[str, object], run.analysis["baseline"])
            variant = _variant_row(run)
            coverage_passed = bool(run.analysis["coverage_passed"])
            baseline_activation_passed = bool(baseline["activation_passed"])
            variant_activation_passed = bool(variant["activation_passed"])
            eligible = bool(
                coverage_passed and baseline_activation_passed and variant_activation_passed
            )
            supports = bool(eligible and variant["supports_variant"])
            support_count += int(supports)
            all_cells_eligible = all_cells_eligible and eligible
            completion_delta = _delta(variant, "completion")
            redirect = cast(dict[str, object], variant["redirect"])
            response_rate = float(cast(float, redirect["response_rate"]))
            completion_deltas.append(completion_delta)
            response_rates.append(response_rate)
            seed_rows.append(
                {
                    "model_family": family,
                    "model": run.model,
                    "seed": run.seed,
                    "coverage_passed": coverage_passed,
                    "baseline_activation_rate": baseline["activation_rate"],
                    "baseline_activation_passed": baseline_activation_passed,
                    "variant_activation_rate": variant["activation_rate"],
                    "variant_activation_passed": variant_activation_passed,
                    "eligible": eligible,
                    "response_rate": response_rate,
                    "completion_rate": variant["completion_rate"],
                    "completion_delta": completion_delta,
                    "prompt_token_delta": _delta(variant, METRIC_PROMPT_TOKENS),
                    "wall_clock_delta_s": _delta(variant, METRIC_WALL_CLOCK),
                    "supports_bilingual": supports,
                    "manifests": run.manifests,
                }
            )
        stable = support_count >= stable_seed_count
        family_rows[family] = {
            "model": family_runs[0].model,
            "seeds": [run.seed for run in family_runs],
            "supported_seeds": support_count,
            "stable_support": stable,
            "routed_feedback_variant": (
                REPEAT_FEEDBACK_BILINGUAL if stable else DEFAULT_REPEAT_FEEDBACK
            ),
            "completion_delta_range": [min(completion_deltas), max(completion_deltas)],
            "response_rate_range": [min(response_rates), max(response_rates)],
        }

    stable_families = [name for name, row in family_rows.items() if row["stable_support"]]
    reference_families = set(cast(list[str], design["reference_model_families"]))
    additional_supported = [name for name in stable_families if name not in reference_families]
    supported_fraction = len(stable_families) / len(families)
    minimum_families = int(cast(int, rule["minimum_supported_families"]))
    minimum_fraction = float(cast(float, rule["minimum_supported_fraction"]))
    global_support = bool(
        all_cells_eligible
        and len(stable_families) >= minimum_families
        and supported_fraction >= minimum_fraction
        and (additional_supported or not rule["require_additional_family_support"])
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
        "supported_family_fraction": supported_fraction,
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
