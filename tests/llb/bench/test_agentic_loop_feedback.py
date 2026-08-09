"""Localized repeat-feedback variants, redirect telemetry, and paired gates."""

import json
from pathlib import Path

from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    REPEATED_ALLOW,
    REPEATED_NOOP,
    REPEATED_NOOP_OBSERVATIONS,
    REPEAT_FEEDBACK_BILINGUAL,
    REPEAT_FEEDBACK_CURRENT,
    REPEAT_FEEDBACK_UK,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_loop_feedback import validate_repeat_feedback_design
from llb.bench.agentic_loop_policy import policy_grid, run_agentic_loop_policy
from llb.bench.agentic_loop_policy_power import load_repeat_power_design


def _design() -> dict[str, object]:
    return {
        "schema_version": 1,
        "study_id": "feedback-test",
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
        "repeat_feedback_variants": [
            REPEAT_FEEDBACK_CURRENT,
            REPEAT_FEEDBACK_UK,
            REPEAT_FEEDBACK_BILINGUAL,
        ],
    }


def _tasks() -> list[AgenticTask]:
    return [
        AgenticTask(
            f"feedback-{index}",
            f"inspect {index}",
            success=[{"kind": "answer_contains", "value": "done"}],
            family="read" if index < 4 else "mutation",
        )
        for index in range(8)
    ]


def _cells():
    return policy_grid(
        [6],
        [MALFORMED_ANSWER],
        [REPEATED_ALLOW, REPEATED_NOOP],
        [REPEAT_FEEDBACK_CURRENT, REPEAT_FEEDBACK_UK, REPEAT_FEEDBACK_BILINGUAL],
    )


def test_feedback_grid_keeps_one_allow_cell_and_all_noop_variants():
    cells = _cells()
    assert len(cells) == 4
    assert [cell.policy.repeat_feedback for cell in cells] == [
        REPEAT_FEEDBACK_CURRENT,
        REPEAT_FEEDBACK_CURRENT,
        REPEAT_FEEDBACK_UK,
        REPEAT_FEEDBACK_BILINGUAL,
    ]
    validate_repeat_feedback_design(_design(), _tasks(), cells=cells, model_family="family-a")


def test_localized_feedback_redirects_and_clears_paired_gates(tmp_path: Path, episode_clock):
    current_notice = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_CURRENT]
    uk_notice = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_UK]
    bilingual_notice = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_BILINGUAL]

    def complete(prompt: str) -> str:
        if uk_notice in prompt or bilingual_notice in prompt:
            return '{"name":"finish","arguments":{"answer":"done"}}'
        if current_notice in prompt:
            return '{"name":"db_get","arguments":{"key":"missing"}}'
        return '{"name":"db_get","arguments":{"key":"missing"}}'

    run = run_agentic_loop_policy(
        _tasks(),
        model="model-a",
        backend="ollama",
        complete=complete,
        max_steps=[6],
        malformed_policies=[MALFORMED_ANSWER],
        repeated_call_policies=[REPEATED_ALLOW, REPEATED_NOOP],
        repeated_feedback_variants=[
            REPEAT_FEEDBACK_CURRENT,
            REPEAT_FEEDBACK_UK,
            REPEAT_FEEDBACK_BILINGUAL,
        ],
        data_dir=tmp_path,
        mirror=lambda *_args: None,
        repeat_feedback_design=_design(),
        model_family="family-a",
        clock=episode_clock(),
    )
    analysis = run.repeat_feedback_analysis
    assert analysis is not None
    assert analysis["coverage_passed"] is True
    assert analysis["baseline"]["redirect"]["response_rate"] == 0.0
    assert analysis["variants"][REPEAT_FEEDBACK_UK]["redirect"]["response_rate"] == 1.0
    uk_redirect = analysis["variants"][REPEAT_FEEDBACK_UK]["redirect"]
    assert uk_redirect["redirected_completion_rate"] == 1.0
    assert uk_redirect["response_completion_rate"] == 1.0
    assert uk_redirect["by_family"]["read"]["completed_redirected_tasks"] == 4
    assert analysis["variants"][REPEAT_FEEDBACK_UK]["supports_variant"] is True
    assert analysis["recommended_feedback_variant"] in {
        REPEAT_FEEDBACK_UK,
        REPEAT_FEEDBACK_BILINGUAL,
    }
    assert run.recommendation["changes_shipped_defaults"] is False
    assert run.recommendation["model_family_supports_feedback_variant"] is True
    assert run.recommendation["model_family_recommended_feedback_variant"] in {
        REPEAT_FEEDBACK_UK,
        REPEAT_FEEDBACK_BILINGUAL,
    }
    assert "cross-family support" in run.recommendation["reason"]
    assert "completion-gate" in run.table and "bilingual" in run.table
    for report in run.reports:
        assert report.paths is not None
        run_dir = Path(report.paths["manifest"]).parent
        persisted = json.loads((run_dir / "feedback-analysis.json").read_text())
        assert persisted["task_family_counts"] == {"mutation": 4, "read": 4}


def test_committed_feedback_design_matches_the_powered_ledger():
    root = Path(__file__).parents[3]
    design = load_repeat_power_design(
        root / "samples/benchmarks/agentic_loop_feedback_localization_design.json"
    )
    from llb.bench.agentic.run import load_tasks_file

    tasks = load_tasks_file(root / "samples/benchmarks/agentic_loop_repeat_power_uk.json")
    validate_repeat_feedback_design(design, tasks, cells=_cells(), model_family="gemma")
