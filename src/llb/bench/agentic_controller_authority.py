"""Prospective observation-versus-controller channel authority comparison."""

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from llb.bench.agentic.controller_channel import (
    CHANNEL_CONTROLLER,
    CHANNEL_OBSERVATION,
)
from llb.bench.agentic_controller_authority_design import (
    CROSS_MODEL_HYPOTHESIS as CROSS_MODEL_HYPOTHESIS,
    CROSS_MODEL_STUDY_KIND as CROSS_MODEL_STUDY_KIND,
    EXPECTED_HYPOTHESIS as EXPECTED_HYPOTHESIS,
    PLACEMENTS,
    STUDY_KIND as STUDY_KIND,
    validate_channel_authority_design as validate_channel_authority_design,
)
from llb.bench.agentic_loop_feedback_outcomes import (
    compact_family_outcomes,
    summarize_response_completion,
)
from llb.core.contracts.benchmarks import AgenticCaseRow
from llb.core.contracts.common import ChatMessage
from llb.rag.fusion_evidence.paired import paired_comparison, reading_of
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets


@dataclass(frozen=True, slots=True)
class ChannelCell:
    placement: str
    rows: list[AgenticCaseRow]
    snapshots: dict[str, list[ChatMessage]]
    manifest: str | None = None
    tokens_per_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ChannelSeedRun:
    seed: int
    model: str
    cells: dict[str, ChannelCell]


def _vector(cell: ChannelCell, metric: str) -> list[float]:
    if metric == "completion":
        return [float(row["success"]) for row in cell.rows]
    return [float(cast(float, row.get(metric, 0.0))) for row in cell.rows]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _snapshot_proof(
    observation: ChannelCell,
    controller: ChannelCell,
    design: dict[str, object],
) -> dict[str, object]:
    task_ids = sorted(observation.snapshots)
    if task_ids != sorted(controller.snapshots):
        raise ValueError("authority activation snapshots differ between placements")
    roster = cast(list[dict[str, object]], design["roster"])
    backend = str(roster[0]["backend"])
    serialization_name = "ollama" if backend == "ollama" else "openai_compatible"
    roles = cast(dict[str, dict[str, str]], design["role_serialization"])[serialization_name]
    pairs: list[dict[str, str]] = []
    for task_id in task_ids:
        baseline = observation.snapshots[task_id]
        candidate = controller.snapshots[task_id]
        if [item["content"] for item in baseline] != [item["content"] for item in candidate]:
            raise ValueError(f"authority snapshot content changed for task {task_id}")
        if baseline[-1]["content"] != design["authority_text"]:
            raise ValueError(f"authority snapshot text is invalid for task {task_id}")
        if baseline[:-1] != candidate[:-1] or baseline[-1]["role"] != roles[CHANNEL_OBSERVATION]:
            raise ValueError(f"observation snapshot structure is invalid for task {task_id}")
        if candidate[-1]["role"] != roles[CHANNEL_CONTROLLER]:
            raise ValueError(f"controller snapshot structure is invalid for task {task_id}")
        normalized = [{**item, "role": "authority"} for item in baseline]
        digest = hashlib.sha256(
            json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        pairs.append({"task_id": task_id, "content_digest": digest})
    return {"passed": True, "paired_tasks": len(pairs), "pairs": pairs}


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


def analyze_channel_authority(
    design: dict[str, object], runs: list[ChannelSeedRun]
) -> dict[str, object]:
    """Apply snapshot, activation, family response, completion, and paired cost gates."""
    seeds = cast(list[int], design["run_seeds"])
    if {run.seed for run in runs} != set(seeds) or len(runs) != len(seeds):
        raise ValueError("controller-channel runs do not match the exact seed grid")
    indexes = bootstrap_index_sets(
        int(cast(int, design["planned_n"])), DEFAULT_RESAMPLES, DEFAULT_SEED
    )
    response_rule = cast(dict[str, object], design["task_family_response_rule"])
    activation_rule = cast(dict[str, object], design["activation_rule"])
    cost_limits = cast(dict[str, float], design["maximum_relative_cost_increase"])
    seed_rows: list[dict[str, object]] = []
    for run in sorted(runs, key=lambda item: item.seed):
        if set(run.cells) != set(PLACEMENTS):
            raise ValueError("controller-channel run does not isolate the two placements")
        baseline = run.cells[CHANNEL_OBSERVATION]
        candidate = run.cells[CHANNEL_CONTROLLER]
        proof = _snapshot_proof(baseline, candidate, design)
        baseline_redirect = summarize_response_completion(baseline.rows)
        redirect = summarize_response_completion(candidate.rows)
        by_family = cast(dict[str, dict[str, object]], redirect["by_family"])
        responsive = [
            family
            for family, row in by_family.items()
            if float(cast(float, row["response_rate"]))
            >= float(cast(float, response_rule["minimum_response_rate"]))
        ]
        activation_passed = all(
            int(cast(int, row["activated_tasks"]))
            >= int(cast(int, activation_rule["minimum_activated_tasks_per_family"]))
            for row in by_family.values()
        ) and int(cast(int, redirect["activated_tasks"])) >= int(
            cast(int, activation_rule["minimum_activated_tasks"])
        )
        completion = paired_comparison(
            _vector(candidate, "completion"), _vector(baseline, "completion"), indexes
        )
        completion_delta = cast(dict[str, float], completion["delta"])
        completion_passed = bool(
            completion_delta["mean"]
            >= float(cast(float, design["minimum_detectable_completion_gain"]))
            and completion["wins"] + completion["losses"]
            >= int(cast(int, design["minimum_discordant_pairs"]))
            and reading_of(completion) == "separated"
        )
        costs = {}
        for metric in ("total_model_input_tokens", "elapsed_s"):
            comparison = paired_comparison(
                _vector(candidate, metric), _vector(baseline, metric), indexes
            )
            costs[metric] = _cost_gate(
                cast(dict[str, object], comparison),
                _mean(_vector(baseline, metric)),
                float(cost_limits[metric]),
            )
        family_passed = len(responsive) >= int(
            cast(int, response_rule["minimum_supported_task_families_per_seed"])
        )
        supports = bool(
            proof["passed"]
            and activation_passed
            and family_passed
            and completion_passed
            and all(cast(dict[str, object], gate)["passed"] for gate in costs.values())
        )
        seed_rows.append(
            {
                "seed": run.seed,
                "model": run.model,
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
                "supports_controller_channel": supports,
                "manifests": {name: cell.manifest for name, cell in run.cells.items()},
            }
        )
    required = int(cast(int, response_rule["minimum_supported_seeds"]))
    supported = sum(bool(row["supports_controller_channel"]) for row in seed_rows)
    return {
        "study_id": design["study_id"],
        "study_kind": design["study_kind"],
        "seed_rows": seed_rows,
        "supported_seeds": supported,
        "required_supported_seeds": required,
        "supports_structural_controller_authority": supported >= required,
        "recommended_placement": (
            CHANNEL_CONTROLLER if supported >= required else CHANNEL_OBSERVATION
        ),
    }
