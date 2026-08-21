"""Prospective observation-versus-controller channel authority comparison."""

from dataclasses import dataclass
from typing import cast

from llb.bench.agentic.controller_channel import (
    CHANNEL_OBSERVATION,
)
from llb.bench.controller_authority.design import (
    CROSS_MODEL_HYPOTHESIS as CROSS_MODEL_HYPOTHESIS,
    CROSS_MODEL_STUDY_KIND as CROSS_MODEL_STUDY_KIND,
    EXPECTED_HYPOTHESIS as EXPECTED_HYPOTHESIS,
    PLACEMENTS,
    PREAMBLE_HYPOTHESIS as PREAMBLE_HYPOTHESIS,
    PREAMBLE_PLACEMENTS,
    PREAMBLE_STUDY_KIND as PREAMBLE_STUDY_KIND,
    STUDY_KIND as STUDY_KIND,
    validate_channel_authority_design as validate_channel_authority_design,
)
from llb.bench.agentic.design_fields import as_float, as_int, as_ints, as_mapping, as_rows, as_str
from llb.bench.loop_feedback.outcomes import (
    compact_family_outcomes,
    summarize_response_completion,
)
from llb.bench.controller_authority.model import ChannelCell, ChannelSeedRun
from llb.bench.controller_authority.snapshot import snapshot_proof
from llb.rag.fusion_evidence.paired import paired_comparison, reading_of
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets


def _vector(cell: ChannelCell, metric: str) -> list[float]:
    if metric == "completion":
        return [float(row["success"]) for row in cell.rows]
    return [float(cast(float, row.get(metric, 0.0))) for row in cell.rows]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cost_gate(comparison: dict[str, object], baseline: float, limit: float) -> dict[str, object]:
    delta = cast(dict[str, float], comparison["delta"])
    allowed = baseline * limit
    return {
        "relative_increase_limit": limit,
        "baseline_mean": baseline,
        "allowed_delta": allowed,
        "paired_delta": delta,
        "passed": delta["hi"] <= allowed,
    }


@dataclass(frozen=True, slots=True)
class _ChannelGates:
    """Everything the design fixes before any seed row is read.

    Built once and handed to each row, so the per-seed reading cannot quietly resolve a threshold
    differently from the row beside it -- and so the row function takes ONE argument instead of the
    six rules it needs.
    """

    candidate_placement: str
    placements: tuple[str, ...]
    families_by_model: dict[str, str]
    indexes: list[list[int]]
    response_rule: dict[str, object]
    activation_rule: dict[str, object]
    cost_limits: dict[str, float]
    minimum_completion_gain: float
    minimum_discordant_pairs: int

    @classmethod
    def from_design(cls, design: dict[str, object]) -> "_ChannelGates":
        """Resolve the design's rules once, in the shape the per-seed reading consumes them."""
        placements = (
            PREAMBLE_PLACEMENTS if design["study_kind"] == PREAMBLE_STUDY_KIND else PLACEMENTS
        )
        roster = as_rows(design, "roster")
        return cls(
            candidate_placement=placements[1],
            placements=tuple(placements),
            families_by_model={as_str(row, "model"): as_str(row, "model_family") for row in roster},
            indexes=bootstrap_index_sets(
                as_int(design, "planned_n"), DEFAULT_RESAMPLES, DEFAULT_SEED
            ),
            response_rule=as_mapping(design, "task_family_response_rule"),
            activation_rule=as_mapping(design, "activation_rule"),
            cost_limits=cast(dict[str, float], design["maximum_relative_cost_increase"]),
            minimum_completion_gain=as_float(design, "minimum_detectable_completion_gain"),
            minimum_discordant_pairs=as_int(design, "minimum_discordant_pairs"),
        )


def _check_seed_grid(design: dict[str, object], runs: list[ChannelSeedRun]) -> None:
    """Every declared model ran on every declared seed, exactly once."""
    expected = {
        (as_str(row, "model"), seed)
        for row in as_rows(design, "roster")
        for seed in as_ints(design, "run_seeds")
    }
    actual = {(run.model, run.seed) for run in runs}
    if actual != expected or len(runs) != len(expected):
        raise ValueError("controller-channel runs do not match the exact seed grid")


def _activation_passed(
    redirect: dict[str, object], by_family: dict[str, dict[str, object]], rule: dict[str, object]
) -> bool:
    """Enough of the ledger actually triggered the notice, per family and overall."""
    per_family = as_int(rule, "minimum_activated_tasks_per_family")
    return all(
        as_int(row, "activated_tasks") >= per_family for row in by_family.values()
    ) and as_int(redirect, "activated_tasks") >= as_int(rule, "minimum_activated_tasks")


def _completion_gate(
    candidate: ChannelCell, baseline: ChannelCell, gates: _ChannelGates
) -> tuple[dict[str, object], bool]:
    """The paired completion reading, and whether it clears the predeclared effect gate."""
    completion = paired_comparison(
        _vector(candidate, "completion"), _vector(baseline, "completion"), gates.indexes
    )
    delta = cast(dict[str, float], completion["delta"])
    passed = bool(
        delta["mean"] >= gates.minimum_completion_gain
        and completion["wins"] + completion["losses"] >= gates.minimum_discordant_pairs
        and reading_of(completion) == "separated"
    )
    return cast(dict[str, object], completion), passed


def _cost_gates(
    candidate: ChannelCell, baseline: ChannelCell, gates: _ChannelGates
) -> dict[str, dict[str, object]]:
    """The paired cost readings the placement may not exceed to be adopted."""
    return {
        metric: _cost_gate(
            cast(
                dict[str, object],
                paired_comparison(
                    _vector(candidate, metric), _vector(baseline, metric), gates.indexes
                ),
            ),
            _mean(_vector(baseline, metric)),
            float(gates.cost_limits[metric]),
        )
        for metric in ("total_model_input_tokens", "elapsed_s")
    }


def _channel_seed_row(
    run: ChannelSeedRun, design: dict[str, object], gates: _ChannelGates
) -> dict[str, object]:
    """One model/seed cell: the snapshot proof, every gate over it, and what it supports."""
    if set(run.cells) != set(gates.placements):
        raise ValueError("controller-channel run does not isolate the two placements")
    baseline = run.cells[CHANNEL_OBSERVATION]
    candidate = run.cells[gates.candidate_placement]
    proof = snapshot_proof(baseline, candidate, design, run.backend)
    baseline_redirect = summarize_response_completion(baseline.rows)
    redirect = summarize_response_completion(candidate.rows)
    by_family = cast(dict[str, dict[str, object]], redirect["by_family"])
    floor = as_float(gates.response_rule, "minimum_response_rate")
    responsive = [
        family for family, row in by_family.items() if as_float(row, "response_rate") >= floor
    ]
    family_passed = len(responsive) >= as_int(
        gates.response_rule, "minimum_supported_task_families_per_seed"
    )
    activation_passed = _activation_passed(redirect, by_family, gates.activation_rule)
    completion, completion_passed = _completion_gate(candidate, baseline, gates)
    costs = _cost_gates(candidate, baseline, gates)
    supports = bool(
        proof["passed"]
        and activation_passed
        and family_passed
        and completion_passed
        and all(gate["passed"] for gate in costs.values())
    )
    return {
        "seed": run.seed,
        "model": run.model,
        "model_family": gates.families_by_model[run.model],
        "snapshot_proof": proof,
        "activation_passed": activation_passed,
        "baseline_response_rate": baseline_redirect["response_rate"],
        "response_rate": redirect["response_rate"],
        "task_family_response_completion": compact_family_outcomes(by_family),
        "responsive_task_families": responsive,
        "task_family_response_gate_passed": family_passed,
        "baseline_completion_rate": _mean(_vector(baseline, "completion")),
        "completion_rate": _mean(_vector(candidate, "completion")),
        "completion_comparison": completion,
        "completion_gate_passed": completion_passed,
        "cost": costs,
        "supports_candidate_placement": supports,
        "supports_controller_channel": supports,
        "manifests": {name: cell.manifest for name, cell in run.cells.items()},
    }


def analyze_channel_authority(
    design: dict[str, object], runs: list[ChannelSeedRun]
) -> dict[str, object]:
    """Apply snapshot, activation, family response, completion, and paired cost gates."""
    _check_seed_grid(design, runs)
    gates = _ChannelGates.from_design(design)
    ordered = sorted(runs, key=lambda item: (gates.families_by_model[item.model], item.seed))
    seed_rows = [_channel_seed_row(run, design, gates) for run in ordered]
    required = as_int(gates.response_rule, "minimum_supported_seeds")
    supported = sum(bool(row["supports_controller_channel"]) for row in seed_rows)
    supports_candidate = supported >= required
    result: dict[str, object] = {
        "study_id": design["study_id"],
        "study_kind": design["study_kind"],
        "seed_rows": seed_rows,
        "supported_seeds": supported,
        "required_supported_seeds": required,
        "supports_candidate_placement": supports_candidate,
        "supports_structural_controller_authority": supports_candidate,
        "recommended_placement": (
            gates.candidate_placement if supports_candidate else CHANNEL_OBSERVATION
        ),
    }
    if design["study_kind"] == PREAMBLE_STUDY_KIND:
        result["supports_template_native_preamble"] = supports_candidate
    return result
