"""Controller-authority feedback contract, decision, and persistence tests."""

import json
from pathlib import Path

import pytest

from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    REPEATED_ALLOW,
    REPEATED_NOOP,
    REPEATED_NOOP_OBSERVATIONS,
    REPEAT_FEEDBACK_GEMMA_AUTHORITY,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.context_policy.run import task_set_digest
from llb.bench.loop_feedback.authority import (
    EXPECTED_HYPOTHESIS,
    FORBIDDEN_NOTICE_TERMS,
    FeedbackAuthorityRun,
    analyze_feedback_authority,
    validate_feedback_authority_design,
)
from llb.bench.loop_feedback.authority_report import (
    format_feedback_authority_table,
    persist_feedback_authority,
)
from llb.bench.loop_policy.power import load_repeat_power_design
from llb.bench.loop_policy.run import run_agentic_loop_policy

FAMILIES = [
    "calculator_holdout",
    "mutation_holdout",
    "read_holdout",
    "search_holdout",
]
SEEDS = [107, 149]


def _tasks() -> list[AgenticTask]:
    return [
        AgenticTask(
            f"authority-{index}",
            f"perform the same operation twice for case {index}, then finish with done",
            success=[{"kind": "answer_contains", "value": "done"}],
            family=FAMILIES[index // 2],
        )
        for index in range(8)
    ]


def _design(tasks: list[AgenticTask]) -> dict[str, object]:
    notice = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_GEMMA_AUTHORITY]
    return {
        "schema_version": 1,
        "study_id": "authority-test",
        "study_kind": "repeat_feedback_controller_authority_transfer",
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
        "repeat_feedback_variants": ["current", REPEAT_FEEDBACK_GEMMA_AUTHORITY],
        "candidate_feedback_variant": REPEAT_FEEDBACK_GEMMA_AUTHORITY,
        "notice_text": notice,
        "maximum_notice_chars": 128,
        "forbidden_notice_terms": list(FORBIDDEN_NOTICE_TERMS),
        "run_seeds": SEEDS,
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


def _analysis(seed: int, *, responsive_families: int = 4) -> dict[str, object]:
    supported = responsive_families >= 3
    completion_delta = 0.5 if supported else 0.0
    completion = {"mean": completion_delta, "lo": completion_delta, "hi": completion_delta}
    cost_delta = {"mean": -8.0, "lo": -10.0, "hi": -6.0}
    cost = {"paired_delta": cost_delta, "passed": True}
    by_family = {
        family: {
            "tasks": 2,
            "activated_tasks": 2,
            "redirected_tasks": 1 if index < responsive_families else 0,
            "completed_redirected_tasks": 1 if index < responsive_families else 0,
            "response_rate": 0.5 if index < responsive_families else 0.0,
            "redirected_completion_rate": 0.5 if index < responsive_families else 0.0,
            "response_completion_rate": 1.0 if index < responsive_families else 0.0,
        }
        for index, family in enumerate(FAMILIES)
    }
    return {
        "model_family": "gemma",
        "run_seed": seed,
        "coverage_passed": True,
        "baseline": {
            "activation_passed": True,
            "redirect": {
                "response_rate": 0.0,
                "by_family": {family: {"response_rate": 0.0} for family in FAMILIES},
            },
        },
        "variants": {
            REPEAT_FEEDBACK_GEMMA_AUTHORITY: {
                "activation_passed": True,
                "supports_variant": supported,
                "completion_rate": 0.75 if supported else 0.25,
                "redirect": {"response_rate": responsive_families / 8, "by_family": by_family},
                "completion": {"paired": {"delta": completion}, "passed": supported},
                "cost": {
                    "total_model_input_tokens": cost,
                    "elapsed_s": cost,
                    "passed": True,
                },
            }
        },
    }


def _runs(*, weak_seed: int | None = None) -> list[FeedbackAuthorityRun]:
    return [
        FeedbackAuthorityRun(
            seed,
            "gemma-model",
            _analysis(seed, responsive_families=2 if seed == weak_seed else 4),
            {REPEAT_FEEDBACK_GEMMA_AUTHORITY: f"/runs/{seed}/manifest.json"},
        )
        for seed in SEEDS
    ]


def test_design_locks_authority_wording_hypothesis_fresh_digest_and_seeds():
    tasks = _tasks()
    design = _design(tasks)
    validate_feedback_authority_design(design, tasks)

    design["notice_text"] += " Search next."
    with pytest.raises(ValueError, match="registered immutable text"):
        validate_feedback_authority_design(design, tasks)
    design = _design(tasks)
    design["reference"]["excluded_prior_task_set_digests"] = [task_set_digest(tasks)]
    with pytest.raises(ValueError, match="must be fresh"):
        validate_feedback_authority_design(design, tasks)


def test_authority_requires_three_families_and_both_seeds(tmp_path: Path):
    design = _design(_tasks())
    analysis = analyze_feedback_authority(design, _runs())
    assert analysis["supports_controller_authority_transfer"] is True
    assert analysis["recommended_feedback_variant"] == REPEAT_FEEDBACK_GEMMA_AUTHORITY
    assert all(row["task_family_response_gate_passed"] for row in analysis["seed_rows"])
    table = format_feedback_authority_table(analysis)
    assert "RCPW" in table and "calculator=0.500" in table

    paths = persist_feedback_authority(
        design,
        analysis,
        data_dir=tmp_path,
        task_digest=task_set_digest(_tasks()),
        table=table,
        mirror=lambda *_args: None,
    )
    run_dir = Path(paths["manifest"]).parent
    persisted = json.loads((run_dir / "controller-authority-transfer-analysis.json").read_text())
    assert persisted["supported_seeds"] == 2
    assert (
        "controller-authority"
        in (run_dir / "controller-authority-transfer-comparison.md").read_text()
    )
    manifest = json.loads(Path(paths["manifest"]).read_text())
    assert manifest["config"]["study_kind"] == "repeat_feedback_controller_authority_transfer"

    weak = analyze_feedback_authority(design, _runs(weak_seed=149))
    assert weak["supports_controller_authority_transfer"] is False
    assert weak["recommended_feedback_variant"] == "current"


def test_loop_runner_applies_only_the_predeclared_authority_candidate(episode_clock):
    tasks = _tasks()
    design = _design(tasks)
    authority_notice = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_GEMMA_AUTHORITY]

    def complete(prompt: str) -> str:
        if authority_notice in prompt:
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
        repeated_feedback_variants=["current", REPEAT_FEEDBACK_GEMMA_AUTHORITY],
        persist=False,
        repeat_feedback_design=design,
        model_family="gemma",
        run_seed=107,
        clock=episode_clock(),
    )
    analysis = run.repeat_feedback_analysis
    assert analysis is not None
    assert set(analysis["variants"]) == {REPEAT_FEEDBACK_GEMMA_AUTHORITY}
    assert analysis["variants"][REPEAT_FEEDBACK_GEMMA_AUTHORITY]["supports_variant"] is True


def test_committed_authority_contract_is_balanced_fresh_and_valid():
    root = Path(__file__).parents[4]
    design = load_repeat_power_design(
        root / "samples/benchmarks/agentic_loop_feedback_controller_authority_design.json"
    )
    from llb.bench.agentic.run import load_tasks_file

    tasks = load_tasks_file(
        root / "samples/benchmarks/agentic_loop_feedback_controller_authority.json"
    )
    validate_feedback_authority_design(design, tasks)
    digest = task_set_digest(tasks)
    assert digest not in design["reference"]["excluded_prior_task_set_digests"]
    assert {family: sum(task.family == family for task in tasks) for family in FAMILIES} == {
        family: 8 for family in FAMILIES
    }
