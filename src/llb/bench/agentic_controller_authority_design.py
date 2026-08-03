"""Prospective design validation for the controller-channel authority lane."""

from collections import Counter
from typing import cast

from llb.bench.agentic.controller_channel import (
    CHANNEL_CONTROLLER,
    CHANNEL_OBSERVATION,
    CHANNEL_PREAMBLE,
    DEFAULT_PREAMBLE_SERIALIZATION,
    DEFAULT_ROLE_SERIALIZATION,
)
from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    REPEATED_NOOP,
    REPEATED_NOOP_OBSERVATIONS,
    REPEAT_FEEDBACK_GEMMA_AUTHORITY,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_context import task_set_digest

STUDY_KIND = "agent_loop_policy_controller_channel_authority"
CROSS_MODEL_STUDY_KIND = "agent_loop_policy_controller_channel_authority_cross_model"
PREAMBLE_STUDY_KIND = "agent_loop_policy_controller_preamble_placement"
EXPECTED_HYPOTHESIS = (
    "A dedicated controller-role authority message will outperform the identical message carried "
    "as an observation across at least three task families while preserving paired cost bounds."
)
CROSS_MODEL_HYPOTHESIS = (
    "A dedicated controller-role authority message will outperform the identical message carried "
    "as an observation on a non-Gemma model family while preserving the original activation, "
    "response, completion, and paired cost gates."
)
PREAMBLE_HYPOTHESIS = (
    "A template-native leading system preamble will outperform the identical authority text "
    "carried as a later observation across the Gemma and Qwen families while preserving the "
    "existing activation, response, completion, and paired cost gates."
)
TRANSFER_REFERENCE = {
    "study_id": "agent-loop-policy-controller-channel-authority-v1",
    "model_family": "gemma",
    "task_set_digest": "5d6148e0851ca749a65fd75768b388931419ae3dccf2c03be20c095c97e9ead1",
}
PREAMBLE_SOURCE_REFERENCE = {
    "source_study_id": "agent-loop-policy-controller-channel-authority-v1",
    "task_set_digest": "5d6148e0851ca749a65fd75768b388931419ae3dccf2c03be20c095c97e9ead1",
}
PREAMBLE_ROSTER = [
    {
        "model_family": "gemma",
        "model": "hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M",
        "backend": "ollama",
    },
    {"model_family": "qwen", "model": "qwen3:14b", "backend": "ollama"},
]
PLACEMENTS = (CHANNEL_OBSERVATION, CHANNEL_CONTROLLER)
PREAMBLE_PLACEMENTS = (CHANNEL_OBSERVATION, CHANNEL_PREAMBLE)
FORBIDDEN_TERMS = (
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


def validate_channel_authority_design(design: dict[str, object], tasks: list[AgenticTask]) -> None:
    """Refuse inference unless text, roles, fresh ledger, seeds, and gates are immutable."""
    study_kind = str(design.get("study_kind", ""))
    expected_hypothesis = {
        STUDY_KIND: EXPECTED_HYPOTHESIS,
        CROSS_MODEL_STUDY_KIND: CROSS_MODEL_HYPOTHESIS,
        PREAMBLE_STUDY_KIND: PREAMBLE_HYPOTHESIS,
    }.get(study_kind)
    if expected_hypothesis is None or design.get("hypothesis") != expected_hypothesis:
        raise ValueError("controller-channel study identity or hypothesis is not immutable")
    reference = cast(dict[str, object], design["reference"])
    digest = task_set_digest(tasks)
    excluded = cast(list[str], reference["excluded_prior_task_set_digests"])
    if reference.get("task_set_digest") != digest:
        raise ValueError("controller-channel task digest does not match the holdout ledger")
    if study_kind == PREAMBLE_STUDY_KIND:
        actual_source = {
            "source_study_id": reference.get("source_study_id"),
            "task_set_digest": reference.get("task_set_digest"),
        }
        if actual_source != PREAMBLE_SOURCE_REFERENCE:
            raise ValueError("controller preamble source ledger is not predeclared exactly")
    elif not excluded or digest in excluded:
        raise ValueError("controller-channel ledger must be fresh relative to prior ledgers")
    planned_n = int(cast(int, design["planned_n"]))
    if len(tasks) != planned_n:
        raise ValueError("controller-channel planned_n does not match the task ledger")
    expected_counts = cast(dict[str, int], design["required_task_families"])
    actual_counts = Counter(task.family or "" for task in tasks)
    if dict(sorted(actual_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError("controller-channel task-family balance does not match the design")

    placements = PREAMBLE_PLACEMENTS if study_kind == PREAMBLE_STUDY_KIND else PLACEMENTS
    if cast(list[str], design["placements"]) != list(placements):
        raise ValueError("controller-channel placements are not predeclared exactly")
    if study_kind == PREAMBLE_STUDY_KIND:
        transforms = design.get("serializer_transforms")
        if transforms != DEFAULT_PREAMBLE_SERIALIZATION:
            raise ValueError(
                "controller preamble serializer transforms are not predeclared exactly"
            )
    else:
        serialization = cast(dict[str, dict[str, str]], design["role_serialization"])
        if serialization != DEFAULT_ROLE_SERIALIZATION:
            raise ValueError(
                "controller-channel backend role serialization is not predeclared exactly"
            )
    notice = cast(str, design["authority_text"])
    expected_notice = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_GEMMA_AUTHORITY]
    if notice != expected_notice or not notice.isascii():
        raise ValueError("controller-channel authority text is not the immutable registered notice")
    if cast(list[str], design["forbidden_terms"]) != list(FORBIDDEN_TERMS):
        raise ValueError("controller-channel forbidden-term contract changed")
    if any(term in notice.casefold() for term in FORBIDDEN_TERMS):
        raise ValueError("controller-channel authority text contains task-specific language")

    seeds = cast(list[int], design["run_seeds"])
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("controller-channel study requires exactly two unique seeds")
    sampling = cast(dict[str, object], design["sampling"])
    if not 0.0 < float(cast(float, sampling["temperature"])) <= 1.0:
        raise ValueError("controller-channel seeded temperature must be in (0, 1]")
    if int(cast(int, sampling["max_tokens"])) <= 0:
        raise ValueError("controller-channel max_tokens must be positive")
    if int(cast(int, design["max_model_len"])) <= 0:
        raise ValueError("controller-channel max_model_len must be positive")
    roster = cast(list[dict[str, object]], design["roster"])
    if not roster or any(not row.get("model") or not row.get("model_family") for row in roster):
        raise ValueError("controller-channel study requires complete backend roster rows")
    if study_kind != PREAMBLE_STUDY_KIND and len(roster) != 1:
        raise ValueError("controller-channel role study requires exactly one backend roster row")
    if study_kind == PREAMBLE_STUDY_KIND:
        if roster != PREAMBLE_ROSTER:
            raise ValueError("controller preamble Gemma/Qwen roster is not predeclared exactly")
    if study_kind == CROSS_MODEL_STUDY_KIND:
        transfer = cast(dict[str, object], design.get("transfer_reference", {}))
        if transfer != TRANSFER_REFERENCE:
            raise ValueError("cross-model controller-channel reference is not immutable")
        if roster[0]["model_family"] == transfer["model_family"]:
            raise ValueError("cross-model controller-channel roster must be non-Gemma")
    for row in roster:
        backend = str(row.get("backend", ""))
        serialization_name = "ollama" if backend == "ollama" else "openai_compatible"
        declared = (
            cast(dict[str, object], design["serializer_transforms"])
            if study_kind == PREAMBLE_STUDY_KIND
            else cast(dict[str, object], design["role_serialization"])
        )
        if serialization_name not in declared:
            raise ValueError("controller-channel roster backend has no serialization")
    fixed = cast(dict[str, object], design["fixed_policy"])
    if fixed != {
        "max_steps": 6,
        "malformed_call": MALFORMED_ANSWER,
        "repeated_call": REPEATED_NOOP,
    }:
        raise ValueError("controller-channel loop policy must remain 6/answer/noop")

    activation = cast(dict[str, int], design["activation_rule"])
    if not 0 < activation["minimum_activated_tasks"] <= planned_n:
        raise ValueError("controller-channel activation total is outside the ledger")
    if activation["minimum_activated_tasks_per_family"] <= 0:
        raise ValueError("controller-channel family activation floor must be positive")
    response = cast(dict[str, object], design["task_family_response_rule"])
    if not 0.0 < float(cast(float, response["minimum_response_rate"])) <= 1.0:
        raise ValueError("controller-channel response floor must be in (0, 1]")
    minimum_families = int(cast(int, response["minimum_supported_task_families_per_seed"]))
    if not 3 <= minimum_families <= len(expected_counts):
        raise ValueError("controller-channel family breadth is outside the ledger")
    required_cells = len(seeds) * (len(roster) if study_kind == PREAMBLE_STUDY_KIND else 1)
    if int(cast(int, response["minimum_supported_seeds"])) != required_cells:
        raise ValueError("controller-channel adoption must require every model-seed cell")
    mde = float(cast(float, design["minimum_detectable_completion_gain"]))
    discordant = int(cast(int, design["minimum_discordant_pairs"]))
    if not 0.0 < mde <= 1.0 or not 4 <= discordant <= planned_n:
        raise ValueError("controller-channel completion gates are outside the ledger")
    costs = cast(dict[str, float], design["maximum_relative_cost_increase"])
    if set(costs) != {"total_model_input_tokens", "elapsed_s"} or any(
        value < 0.0 for value in costs.values()
    ):
        raise ValueError("controller-channel cost gates are invalid")
