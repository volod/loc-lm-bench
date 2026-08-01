"""Prospective Gemma task-family transfer contract and seeded gate tests."""

import json
from pathlib import Path

import pytest

from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    REPEATED_ALLOW,
    REPEATED_NOOP,
    REPEATED_NOOP_OBSERVATIONS,
    REPEAT_FEEDBACK_GEMMA_PROGRESS,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_context import task_set_digest
from llb.bench.agentic_loop_feedback_transfer import (
    EXPECTED_HYPOTHESIS,
    FORBIDDEN_NOTICE_TERMS,
    FeedbackTransferRun,
    analyze_feedback_transfer,
    validate_feedback_transfer_design,
)
from llb.bench.agentic_loop_feedback_transfer_report import (
    format_feedback_transfer_table,
    persist_feedback_transfer,
)
from llb.bench.agentic_loop_policy import run_agentic_loop_policy
from llb.bench.agentic_loop_policy_power import load_repeat_power_design

FAMILIES = [
    "calculator_holdout",
    "mutation_holdout",
    "read_holdout",
    "search_holdout",
]


def _tasks() -> list[AgenticTask]:
    return [
        AgenticTask(
            f"transfer-{index}",
            f"perform the same operation twice for case {index}, then finish with done",
            success=[{"kind": "answer_contains", "value": "done"}],
            family=FAMILIES[index // 2],
        )
        for index in range(8)
    ]


def _design(tasks: list[AgenticTask]) -> dict[str, object]:
    notice = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_GEMMA_PROGRESS]
    return {
        "schema_version": 1,
        "study_id": "transfer-test",
        "study_kind": "repeat_feedback_task_family_transfer",
        "hypothesis": EXPECTED_HYPOTHESIS,
        "reference": {
            "task_set_digest": task_set_digest(tasks),
            "excluded_prior_task_set_digests": ["prior-ledger-digest"],
        },
        "planned_n": 8,
        "minimum_detectable_completion_gain": 0.5,
        "minimum_discordant_pairs": 4,
        "required_task_families": {family: 2 for family in FAMILIES},
        "minimum_activation_rate": 0.5,
        "minimum_activated_tasks_per_family": 1,
        "maximum_relative_cost_increase": {
            "total_model_input_tokens": 0.1,
            "elapsed_s": 0.2,
        },
        "required_model_families": ["gemma"],
        "roster": [{"model_family": "gemma", "model": "gemma-model", "backend": "ollama"}],
        "repeat_feedback_variants": ["current", REPEAT_FEEDBACK_GEMMA_PROGRESS],
        "candidate_feedback_variant": REPEAT_FEEDBACK_GEMMA_PROGRESS,
        "notice_text": notice,
        "maximum_notice_chars": 120,
        "forbidden_notice_terms": list(FORBIDDEN_NOTICE_TERMS),
        "run_seeds": [41, 73],
        "sampling": {"temperature": 0.2},
        "task_family_response_rule": {
            "minimum_response_rate": 0.25,
            "minimum_supported_task_families_per_seed": 3,
            "minimum_supported_seeds": 2,
        },
        "fixed_policy": {
            "max_steps": 6,
            "malformed_call": "answer",
            "repeated_call": ["allow", "noop"],
        },
    }


def _analysis(seed: int, *, responsive_families: int = 4, supports: bool = True):
    completion_delta = 0.5 if supports else 0.0
    completion = {"mean": completion_delta, "lo": completion_delta, "hi": completion_delta}
    cost_delta = {"mean": -10.0, "lo": -12.0, "hi": -8.0}
    cost = {"paired_delta": cost_delta, "passed": True}
    by_family = {
        family: {
            "tasks": 2,
            "activated_tasks": 2,
            "redirected_tasks": 1 if index < responsive_families else 0,
            "response_rate": 0.5 if index < responsive_families else 0.0,
        }
        for index, family in enumerate(FAMILIES)
    }
    return {
        "model_family": "gemma",
        "run_seed": seed,
        "coverage_passed": True,
        "baseline": {
            "activation_rate": 1.0,
            "activation_passed": True,
            "redirect": {
                "response_rate": 0.0,
                "by_family": {family: {"response_rate": 0.0} for family in FAMILIES},
            },
        },
        "variants": {
            REPEAT_FEEDBACK_GEMMA_PROGRESS: {
                "activation_passed": True,
                "activation_rate": 1.0,
                "supports_variant": supports,
                "completion_rate": 0.75 if supports else 0.25,
                "redirect": {
                    "response_rate": responsive_families / len(FAMILIES) / 2,
                    "by_family": by_family,
                },
                "completion": {"paired": {"delta": completion}, "passed": supports},
                "cost": {
                    "total_model_input_tokens": cost,
                    "elapsed_s": cost,
                    "passed": True,
                },
            }
        },
    }


def _runs(*, weak_seed: int | None = None) -> list[FeedbackTransferRun]:
    return [
        FeedbackTransferRun(
            seed,
            "gemma-model",
            _analysis(seed, responsive_families=2 if seed == weak_seed else 4),
            {"gemma_progress": f"/runs/{seed}/manifest.json"},
        )
        for seed in [41, 73]
    ]


def test_design_locks_neutral_notice_hypothesis_fresh_digest_and_seed_grid():
    tasks = _tasks()
    design = _design(tasks)
    validate_feedback_transfer_design(design, tasks)

    design["notice_text"] += " Search again."
    with pytest.raises(ValueError, match="registered immutable text"):
        validate_feedback_transfer_design(design, tasks)
    design = _design(tasks)
    design["reference"]["excluded_prior_task_set_digests"] = [task_set_digest(tasks)]
    with pytest.raises(ValueError, match="must be fresh"):
        validate_feedback_transfer_design(design, tasks)


def test_transfer_requires_response_in_three_families_and_support_on_both_seeds(
    tmp_path: Path,
):
    design = _design(_tasks())
    analysis = analyze_feedback_transfer(design, _runs())
    assert analysis["supports_task_family_transfer"] is True
    assert analysis["recommended_feedback_variant"] == REPEAT_FEEDBACK_GEMMA_PROGRESS
    assert all(row["task_family_response_gate_passed"] for row in analysis["seed_rows"])
    assert analysis["seed_rows"][0]["response_rate_delta"] == pytest.approx(0.5)
    assert analysis["seed_rows"][0]["completion_comparison"]["delta"]["mean"] == 0.5
    table = format_feedback_transfer_table(analysis)
    assert "RCPW" in table and "calculator=0.500" in table

    paths = persist_feedback_transfer(
        design,
        analysis,
        data_dir=tmp_path,
        task_digest=task_set_digest(_tasks()),
        table=table,
        mirror=lambda *_args: None,
    )
    persisted = json.loads(
        (Path(paths["manifest"]).parent / "task-family-transfer-analysis.json").read_text()
    )
    assert persisted["supported_seeds"] == 2

    weak = analyze_feedback_transfer(design, _runs(weak_seed=73))
    assert weak["supports_task_family_transfer"] is False
    assert weak["recommended_feedback_variant"] == "current"


def test_loop_runner_applies_only_the_predeclared_neutral_candidate():
    tasks = _tasks()
    design = _design(tasks)
    candidate_notice = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_GEMMA_PROGRESS]

    def complete(prompt: str) -> str:
        if candidate_notice in prompt:
            return '{"name":"finish","arguments":{"answer":"done"}}'
        return '{"name":"db_get","arguments":{"key":"missing"}}'

    run = run_agentic_loop_policy(
        tasks,
        model="gemma-model",
        backend="ollama",
        complete=complete,
        max_steps=[6],
        malformed_policies=[MALFORMED_ANSWER],
        repeated_call_policies=[REPEATED_ALLOW, REPEATED_NOOP],
        repeated_feedback_variants=["current", REPEAT_FEEDBACK_GEMMA_PROGRESS],
        persist=False,
        repeat_feedback_design=design,
        model_family="gemma",
        run_seed=41,
    )
    analysis = run.repeat_feedback_analysis
    assert analysis is not None
    assert set(analysis["variants"]) == {REPEAT_FEEDBACK_GEMMA_PROGRESS}
    assert analysis["variants"][REPEAT_FEEDBACK_GEMMA_PROGRESS]["supports_variant"] is True


def test_committed_transfer_contract_is_balanced_fresh_and_valid():
    root = Path(__file__).parents[3]
    design = load_repeat_power_design(
        root / "samples/benchmarks/agentic_loop_feedback_task_family_transfer_design.json"
    )
    from llb.bench.agentic.run import load_tasks_file

    tasks = load_tasks_file(
        root / "samples/benchmarks/agentic_loop_feedback_task_family_transfer.json"
    )
    validate_feedback_transfer_design(design, tasks)
    assert task_set_digest(tasks) not in design["reference"]["excluded_prior_task_set_digests"]
    assert {family: sum(task.family == family for task in tasks) for family in FAMILIES} == {
        family: 8 for family in FAMILIES
    }
