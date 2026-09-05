"""Prospective coverage and realized gates for repeat/no-op power runs."""

import json
from pathlib import Path

import pytest

from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    REPEATED_ALLOW,
    REPEATED_NOOP,
    REPEATED_NOOP_OBSERVATION,
    LoopPolicy,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.loop_policy.run import run_agentic_loop_policy
from llb.bench.loop_policy.power import (
    load_repeat_power_design,
    validate_repeat_power_design,
)
from llb.bench.agentic.run import load_tasks_file
from llb.bench.loop_policy.report import LoopPolicyCell


def _design() -> dict[str, object]:
    return {
        "schema_version": 1,
        "study_id": "test-power",
        "planned_n": 8,
        "minimum_detectable_completion_gain": 0.5,
        "minimum_discordant_pairs": 4,
        "required_task_families": {"read": 4, "mutation": 4},
        "minimum_activation_rate": 0.5,
        "minimum_activated_tasks_per_family": 2,
        "maximum_relative_cost_increase": {
            "total_model_input_tokens": 0.1,
            "elapsed_s": 10.0,
        },
        "required_model_families": ["family-a", "family-b"],
    }


def _tasks() -> list[AgenticTask]:
    return [
        AgenticTask(
            f"task-{index}",
            f"inspect {index}",
            success=[{"kind": "answer_contains", "value": "done"}],
            family="read" if index < 4 else "mutation",
        )
        for index in range(8)
    ]


def _cells() -> list[LoopPolicyCell]:
    return [
        LoopPolicyCell(6, LoopPolicy(MALFORMED_ANSWER, REPEATED_ALLOW)),
        LoopPolicyCell(6, LoopPolicy(MALFORMED_ANSWER, REPEATED_NOOP)),
    ]


def test_design_refuses_missing_family_coverage_and_duplicate_payloads():
    tasks = _tasks()
    validate_repeat_power_design(_design(), tasks, cells=_cells(), model_family="family-a")
    duplicate = AgenticTask(
        "other-id",
        tasks[0].prompt,
        setup=tasks[0].setup,
        success=tasks[0].success,
        family=tasks[0].family,
    )
    with pytest.raises(ValueError, match="non-duplicate"):
        validate_repeat_power_design(
            _design(),
            [duplicate, tasks[0], *tasks[2:]],
            cells=_cells(),
            model_family="family-a",
        )


def test_power_run_reports_activation_separately_and_gates_recommendation(tmp_path: Path):
    def complete(prompt: str) -> str:
        if REPEATED_NOOP_OBSERVATION in prompt:
            return '{"name":"finish","arguments":{"answer":"done"}}'
        return '{"name":"db_get","arguments":{"key":"missing"}}'

    run = run_agentic_loop_policy(
        _tasks(),
        model="model-a",
        backend="ollama",
        complete=complete,
        max_steps=[6],
        malformed_policies=[MALFORMED_ANSWER],
        repeated_call_policies=[REPEATED_ALLOW, REPEATED_NOOP],
        data_dir=tmp_path,
        mirror=lambda *_args: None,
        repeat_power_design=_design(),
        model_family="family-a",
    )
    analysis = run.repeat_power_analysis
    assert analysis is not None
    assert analysis["activation"]["allow"]["activation_rate"] == 1.0
    assert analysis["completion"]["paired_delta"]["mean"] == 1.0
    assert analysis["supports_noop"] is True
    assert run.recommendation["model_family_supports_noop"] is True
    assert run.recommendation["changes_shipped_defaults"] is False
    assert "active" in run.table and "repeats" in run.table
    for report in run.reports:
        assert all(row["task_family"] in {"read", "mutation"} for row in report.rows)
        run_dir = Path(report.paths["manifest"]).parent
        persisted = json.loads((run_dir / "power-analysis.json").read_text())
        assert persisted["task_family_counts"] == {"mutation": 4, "read": 4}


def test_committed_power_fixture_matches_its_prospective_design():
    root = Path(__file__).parents[4]
    tasks = load_tasks_file(root / "samples/benchmarks/agentic_loop_repeat_power_uk.json")
    design = load_repeat_power_design(
        root / "samples/benchmarks/agentic_loop_repeat_power_design.json"
    )
    validate_repeat_power_design(design, tasks, cells=_cells(), model_family="gemma")
    assert len(tasks) == 32
