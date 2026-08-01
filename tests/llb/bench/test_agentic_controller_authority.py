"""Structural controller-channel authority contracts and end-to-end fake evidence."""

import json
from pathlib import Path

import pytest

from llb.bench.agentic.context_budget import unbounded_budget
from llb.bench.agentic.controller_channel import (
    CHANNEL_CONTROLLER,
    CHANNEL_OBSERVATION,
    DEFAULT_ROLE_SERIALIZATION,
    ControllerFeedback,
    serialize_controller_transcript,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic.run import load_tasks_file
from llb.bench.agentic_context import task_set_digest
from llb.bench.agentic_controller_authority import (
    EXPECTED_HYPOTHESIS,
    STUDY_KIND,
    analyze_channel_authority,
    validate_channel_authority_design,
)
from llb.bench.agentic_controller_authority_report import (
    format_channel_authority_table,
    persist_channel_authority,
)
from llb.bench.agentic_controller_authority_run import run_channel_authority_seed

AUTHORITY = (
    "[loop] Controller ruling: suppression satisfies the requested repetition. "
    "You must now take the next distinct action."
)
FAMILIES = ["calculator_holdout", "mutation_holdout", "read_holdout", "search_holdout"]


def _tasks() -> list[AgenticTask]:
    return [
        AgenticTask(
            id=f"channel-{index}",
            prompt=f"repeat the lookup for case {index}, then finish with done",
            success=[{"kind": "answer_contains", "value": "done"}],
            family=FAMILIES[index // 2],
        )
        for index in range(8)
    ]


def _design(tasks: list[AgenticTask]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "study_id": "channel-test",
        "study_kind": STUDY_KIND,
        "hypothesis": EXPECTED_HYPOTHESIS,
        "reference": {
            "task_set_digest": task_set_digest(tasks),
            "excluded_prior_task_set_digests": ["prior-digest"],
        },
        "planned_n": 8,
        "required_task_families": {family: 2 for family in FAMILIES},
        "placements": [CHANNEL_OBSERVATION, CHANNEL_CONTROLLER],
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
        "role_serialization": DEFAULT_ROLE_SERIALIZATION,
        "roster": [{"model_family": "gemma", "model": "gemma", "backend": "ollama"}],
        "run_seeds": [211, 257],
        "sampling": {"temperature": 0.2, "max_tokens": 512},
        "fixed_policy": {"max_steps": 6, "malformed_call": "answer", "repeated_call": "noop"},
        "activation_rule": {
            "minimum_activated_tasks": 4,
            "minimum_activated_tasks_per_family": 1,
        },
        "task_family_response_rule": {
            "minimum_response_rate": 0.25,
            "minimum_supported_task_families_per_seed": 3,
            "minimum_supported_seeds": 2,
        },
        "minimum_detectable_completion_gain": 0.5,
        "minimum_discordant_pairs": 4,
        "maximum_relative_cost_increase": {
            "total_model_input_tokens": 1.0,
            "elapsed_s": 100.0,
        },
        "max_model_len": 8192,
    }


def test_serializer_changes_only_the_authority_role():
    observation = serialize_controller_transcript(
        "fixed prompt",
        [ControllerFeedback(AUTHORITY, CHANNEL_OBSERVATION)],
        backend="ollama",
    )
    controller = serialize_controller_transcript(
        "fixed prompt",
        [ControllerFeedback(AUTHORITY, CHANNEL_CONTROLLER)],
        backend="ollama",
    )
    assert [message["content"] for message in observation] == [
        message["content"] for message in controller
    ]
    assert observation[0] == controller[0]
    assert observation[-1]["role"] == "user"
    assert controller[-1]["role"] == "system"


def test_design_locks_fresh_digest_roles_seeds_and_balance():
    tasks = _tasks()
    design = _design(tasks)
    validate_channel_authority_design(design, tasks)
    design["role_serialization"] = {"ollama": {"observation": "tool", "controller": "system"}}
    with pytest.raises(ValueError, match="role serialization"):
        validate_channel_authority_design(design, tasks)
    design = _design(tasks)
    design["reference"]["excluded_prior_task_set_digests"] = [task_set_digest(tasks)]
    with pytest.raises(ValueError, match="fresh"):
        validate_channel_authority_design(design, tasks)


def test_fake_run_proves_snapshots_and_all_adoption_gates(tmp_path: Path):
    tasks = _tasks()
    design = _design(tasks)

    def chat(messages: list[dict[str, str]]) -> str:
        authority = next((message for message in messages if message["content"] == AUTHORITY), None)
        if authority is not None and authority["role"] == "system":
            return '{"name":"finish","arguments":{"answer":"done"}}'
        return '{"name":"db_get","arguments":{"key":"missing"}}'

    runs = [
        run_channel_authority_seed(
            tasks,
            seed=seed,
            model="gemma",
            backend="ollama",
            chat=chat,
            budget=unbounded_budget(),
            max_steps=6,
            design=design,
        )
        for seed in [211, 257]
    ]
    analysis = analyze_channel_authority(design, runs)
    assert analysis["supports_structural_controller_authority"] is True
    assert analysis["recommended_placement"] == CHANNEL_CONTROLLER
    assert all(row["snapshot_proof"]["paired_tasks"] == 8 for row in analysis["seed_rows"])
    assert "SARC" in format_channel_authority_table(analysis)
    paths = persist_channel_authority(
        design,
        analysis,
        data_dir=tmp_path,
        table=format_channel_authority_table(analysis),
        mirror=lambda *_args: None,
    )
    run_dir = Path(paths["manifest"]).parent
    persisted = json.loads((run_dir / "controller-channel-authority-analysis.json").read_text())
    assert persisted["supported_seeds"] == 2


def test_committed_controller_channel_contract_is_fresh_balanced_and_valid():
    root = Path(__file__).parents[3]
    tasks = load_tasks_file(root / "samples/benchmarks/agentic_controller_channel_authority.json")
    design = json.loads(
        (root / "samples/benchmarks/agentic_controller_channel_authority_design.json").read_text()
    )
    validate_channel_authority_design(design, tasks)
    assert task_set_digest(tasks) not in design["reference"]["excluded_prior_task_set_digests"]
    assert {family: sum(task.family == family for task in tasks) for family in FAMILIES} == {
        family: 8 for family in FAMILIES
    }
