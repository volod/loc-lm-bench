"""Template-native controller preamble placement contracts."""

import json
from pathlib import Path

import pytest

from llb.backends.context_budget import unbounded_budget
from llb.bench.agentic.controller_channel import (
    CHANNEL_OBSERVATION,
    CHANNEL_PREAMBLE,
    DEFAULT_PREAMBLE_SERIALIZATION,
    ControllerFeedback,
    serialize_controller_transcript,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic.run import load_tasks_file
from llb.bench.context_policy.run import task_set_digest
from llb.bench.controller_authority.run import (
    PREAMBLE_HYPOTHESIS,
    PREAMBLE_STUDY_KIND,
    analyze_channel_authority,
    validate_channel_authority_design,
)
from llb.bench.controller_authority.episodes import run_channel_authority_seed

AUTHORITY = (
    "[loop] Controller ruling: suppression satisfies the requested repetition. "
    "You must now take the next distinct action."
)
FAMILIES = ["calculator_holdout", "mutation_holdout", "read_holdout", "search_holdout"]


def _tasks() -> list[AgenticTask]:
    return [
        AgenticTask(
            id=f"preamble-{index}",
            prompt=f"repeat the lookup for case {index}, then finish with done",
            success=[{"kind": "answer_contains", "value": "done"}],
            family=FAMILIES[index // 2],
        )
        for index in range(8)
    ]


def _design(tasks: list[AgenticTask]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "study_id": "preamble-test",
        "study_kind": PREAMBLE_STUDY_KIND,
        "hypothesis": PREAMBLE_HYPOTHESIS,
        "reference": {
            "task_set_digest": task_set_digest(tasks),
            "excluded_prior_task_set_digests": ["prior-digest"],
        },
        "planned_n": 8,
        "required_task_families": {family: 2 for family in FAMILIES},
        "placements": [CHANNEL_OBSERVATION, CHANNEL_PREAMBLE],
        "authority_text": AUTHORITY,
        "forbidden_terms": [
            "answer",
            "calculator",
            "database",
            "file",
            "mutation",
            "read",
            "search",
            "tool",
            "write",
        ],
        "serializer_transforms": DEFAULT_PREAMBLE_SERIALIZATION,
        "roster": [
            {"model_family": "gemma", "model": "gemma", "backend": "ollama"},
            {"model_family": "qwen", "model": "qwen", "backend": "ollama"},
        ],
        "run_seeds": [401, 443],
        "sampling": {"temperature": 0.2, "max_tokens": 512},
        "fixed_policy": {"max_steps": 6, "malformed_call": "answer", "repeated_call": "noop"},
        "activation_rule": {
            "minimum_activated_tasks": 4,
            "minimum_activated_tasks_per_family": 1,
        },
        "task_family_response_rule": {
            "minimum_response_rate": 0.25,
            "minimum_supported_task_families_per_seed": 3,
            "minimum_supported_seeds": 4,
        },
        "minimum_detectable_completion_gain": 0.5,
        "minimum_discordant_pairs": 4,
        "maximum_relative_cost_increase": {
            "total_model_input_tokens": 1.0,
            "elapsed_s": 100.0,
        },
        "max_model_len": 8192,
    }


@pytest.mark.parametrize("backend", ["ollama", "vllm"])
def test_serializer_moves_unchanged_authority_to_leading_system_preamble(backend: str):
    observation = serialize_controller_transcript(
        "fixed prompt",
        [ControllerFeedback(AUTHORITY, CHANNEL_OBSERVATION)],
        backend=backend,
        serializer_transforms=DEFAULT_PREAMBLE_SERIALIZATION,
    )
    preamble = serialize_controller_transcript(
        "fixed prompt",
        [ControllerFeedback(AUTHORITY, CHANNEL_PREAMBLE)],
        backend=backend,
        serializer_transforms=DEFAULT_PREAMBLE_SERIALIZATION,
    )

    assert observation == [
        {"role": "user", "content": "fixed prompt"},
        {"role": "user", "content": AUTHORITY},
    ]
    assert preamble == [
        {"role": "system", "content": AUTHORITY},
        {"role": "user", "content": "fixed prompt"},
    ]


def test_fake_two_family_run_proves_reorder_and_all_gates():
    tasks = _tasks()
    design = _design(tasks)

    def chat(messages: list[dict[str, str]]) -> str:
        if messages[0] == {"role": "system", "content": AUTHORITY}:
            return '{"name":"finish","arguments":{"answer":"done"}}'
        return '{"name":"db_get","arguments":{"key":"missing"}}'

    runs = [
        run_channel_authority_seed(
            tasks,
            seed=seed,
            model=model,
            backend="ollama",
            chat=chat,
            budget=unbounded_budget(),
            max_steps=6,
            design=design,
        )
        for model in ["gemma", "qwen"]
        for seed in [401, 443]
    ]
    analysis = analyze_channel_authority(design, runs)

    assert analysis["supports_structural_controller_authority"] is True
    assert analysis["supports_candidate_placement"] is True
    assert analysis["supports_template_native_preamble"] is True
    assert analysis["recommended_placement"] == CHANNEL_PREAMBLE
    assert analysis["supported_seeds"] == 4
    assert all(row["supports_candidate_placement"] for row in analysis["seed_rows"])
    assert {row["model_family"] for row in analysis["seed_rows"]} == {"gemma", "qwen"}

    runs[0].cells[CHANNEL_PREAMBLE].snapshots[tasks[0].id][0]["content"] += " changed"
    with pytest.raises(ValueError, match="authority snapshot content changed"):
        analyze_channel_authority(design, runs)


def test_committed_preamble_contract_pins_transforms_models_and_seeds():
    root = Path(__file__).parents[4]
    tasks = load_tasks_file(root / "samples/benchmarks/agentic_controller_channel_authority.json")
    design = json.loads(
        (root / "samples/benchmarks/agentic_controller_preamble_placement_design.json").read_text()
    )

    validate_channel_authority_design(design, tasks)
    assert [row["model_family"] for row in design["roster"]] == ["gemma", "qwen"]
    assert design["run_seeds"] == [401, 443]
    assert design["serializer_transforms"] == DEFAULT_PREAMBLE_SERIALIZATION

    design["serializer_transforms"]["ollama"]["preamble"].reverse()
    with pytest.raises(ValueError, match="serializer transforms"):
        validate_channel_authority_design(design, tasks)

    design = json.loads(
        (root / "samples/benchmarks/agentic_controller_preamble_placement_design.json").read_text()
    )
    design["roster"][1]["model"] = "qwen3:30b"
    with pytest.raises(ValueError, match="roster"):
        validate_channel_authority_design(design, tasks)
