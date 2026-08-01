"""Prospective wording, seeded family routing, and candidate-isolation checks."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    REPEATED_ALLOW,
    REPEATED_NOOP,
    REPEATED_NOOP_OBSERVATIONS,
    REPEAT_FEEDBACK_AYA_DIRECT,
    REPEAT_FEEDBACK_GEMMA_CHOICE,
    REPEAT_FEEDBACK_MISTRAL_USE,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_context import task_set_digest
from llb.bench.agentic_loop_feedback_adaptation import (
    FeedbackAdaptationRun,
    analyze_feedback_adaptation,
    validate_feedback_adaptation_design,
)
from llb.bench.agentic_loop_feedback_adaptation_report import (
    format_feedback_adaptation_table,
    persist_feedback_adaptation,
)
from llb.bench.agentic_loop_policy import run_agentic_loop_policy
from llb.bench.agentic_loop_policy_power import load_repeat_power_design

CANDIDATES = {
    "aya": REPEAT_FEEDBACK_AYA_DIRECT,
    "mistral": REPEAT_FEEDBACK_MISTRAL_USE,
    "gemma": REPEAT_FEEDBACK_GEMMA_CHOICE,
}


def _tasks() -> list[AgenticTask]:
    return [
        AgenticTask(
            f"adapt-{index}",
            f"inspect {index}",
            success=[{"kind": "answer_contains", "value": "done"}],
            family="read" if index < 4 else "mutation",
        )
        for index in range(8)
    ]


def _design(tasks: list[AgenticTask]) -> dict[str, object]:
    families = list(CANDIDATES)
    candidates = list(CANDIDATES.values())
    return {
        "schema_version": 1,
        "study_id": "adaptation-test",
        "study_kind": "repeat_feedback_family_adaptation",
        "reference": {"task_set_digest": task_set_digest(tasks)},
        "planned_n": 8,
        "minimum_detectable_completion_gain": 0.5,
        "minimum_discordant_pairs": 4,
        "required_task_families": {"read": 4, "mutation": 4},
        "minimum_activation_rate": 0.5,
        "minimum_activated_tasks_per_family": 2,
        "maximum_relative_cost_increase": {
            "total_model_input_tokens": 0.1,
            "elapsed_s": 0.2,
        },
        "required_model_families": families,
        "roster": [
            {
                "model_family": family,
                "model": f"{family}-model",
                "backend": "ollama",
                "candidate_feedback_variant": CANDIDATES[family],
            }
            for family in families
        ],
        "repeat_feedback_variants": ["current", *candidates],
        "notice_text": {name: REPEATED_NOOP_OBSERVATIONS[name] for name in candidates},
        "candidate_hypotheses": {name: f"fixed hypothesis for {name}" for name in candidates},
        "maximum_notice_chars": 120,
        "run_seeds": [13, 29],
        "sampling": {"temperature": 0.2},
        "cross_family_adoption_rule": {
            "minimum_supported_seeds_per_family": 2,
            "minimum_supported_families": 2,
            "minimum_supported_fraction": 2 / 3,
        },
        "fixed_policy": {
            "max_steps": 6,
            "malformed_call": "answer",
            "repeated_call": ["allow", "noop"],
        },
    }


def _analysis(family: str, seed: int, supports: bool) -> dict[str, object]:
    candidate = CANDIDATES[family]
    completion_delta = 0.5 if supports else 0.0
    completion = {
        "mean": completion_delta,
        "lo": completion_delta,
        "hi": completion_delta,
    }
    cost_delta = {"mean": -10.0, "lo": -12.0, "hi": -8.0}
    cost = {"paired_delta": cost_delta, "passed": True}
    return {
        "model_family": family,
        "run_seed": seed,
        "coverage_passed": True,
        "baseline": {"activation_rate": 0.75, "activation_passed": True},
        "variants": {
            candidate: {
                "activation_passed": True,
                "activation_rate": 0.75,
                "supports_variant": supports,
                "completion_rate": 0.75 if supports else 0.25,
                "redirect": {"response_rate": 0.75 if supports else 0.25},
                "completion": {"paired": {"delta": completion}, "passed": supports},
                "cost": {
                    "total_model_input_tokens": cost,
                    "elapsed_s": cost,
                    "passed": True,
                },
            }
        },
    }


def _runs(supported: set[str]) -> list[FeedbackAdaptationRun]:
    return [
        FeedbackAdaptationRun(
            family,
            f"{family}-model",
            seed,
            CANDIDATES[family],
            _analysis(family, seed, family in supported),
            {CANDIDATES[family]: f"/runs/{family}/{seed}/manifest.json"},
        )
        for family in CANDIDATES
        for seed in [13, 29]
    ]


def test_design_predeclares_exact_registered_ascii_notices_and_hypotheses():
    tasks = _tasks()
    design = _design(tasks)
    validate_feedback_adaptation_design(design, tasks)
    design["notice_text"][REPEAT_FEEDBACK_AYA_DIRECT] += " changed after inference"
    with pytest.raises(ValueError, match="does not match"):
        validate_feedback_adaptation_design(design, tasks)


def test_stable_routes_require_both_seeds_and_cross_family_threshold(tmp_path: Path):
    tasks = _tasks()
    design = _design(tasks)
    analysis = analyze_feedback_adaptation(design, _runs({"aya", "mistral"}))
    assert analysis["candidate_isolation_passed"] is True
    assert analysis["stable_supported_families"] == ["aya", "mistral"]
    assert analysis["supports_family_adapted_routing"] is True
    assert analysis["families"]["aya"]["routed_feedback_variant"] == "aya_direct"
    assert analysis["families"]["gemma"]["routed_feedback_variant"] == "current"
    assert analysis["seed_rows"][0]["completion_gate_passed"] is True
    assert analysis["seed_rows"][0]["prompt_cost_gate_passed"] is True
    assert analysis["seed_rows"][0]["wall_cost_gate_passed"] is True
    table = format_feedback_adaptation_table(analysis)
    assert "stable family routing" in table and "gemma_choice" in table and "CPW" in table

    paths = persist_feedback_adaptation(
        design,
        analysis,
        data_dir=tmp_path,
        task_digest=task_set_digest(tasks),
        table=table,
        mirror=lambda *_args: None,
    )
    run_dir = Path(paths["manifest"]).parent
    persisted = json.loads((run_dir / "family-adaptation-analysis.json").read_text())
    assert persisted["supported_family_fraction"] == pytest.approx(2 / 3)


def test_runner_resolves_only_the_candidate_declared_for_its_family():
    tasks = _tasks()
    design = _design(tasks)
    candidate_notice = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_AYA_DIRECT]

    def complete(prompt: str) -> str:
        if candidate_notice in prompt:
            return '{"name":"finish","arguments":{"answer":"done"}}'
        return '{"name":"db_get","arguments":{"key":"missing"}}'

    run = run_agentic_loop_policy(
        tasks,
        model="aya-model",
        backend="ollama",
        complete=complete,
        max_steps=[6],
        malformed_policies=[MALFORMED_ANSWER],
        repeated_call_policies=[REPEATED_ALLOW, REPEATED_NOOP],
        repeated_feedback_variants=["current", REPEAT_FEEDBACK_AYA_DIRECT],
        persist=False,
        repeat_feedback_design=design,
        model_family="aya",
        run_seed=13,
    )
    assert run.repeat_feedback_analysis is not None
    assert set(run.repeat_feedback_analysis["variants"]) == {REPEAT_FEEDBACK_AYA_DIRECT}
    assert (
        run.repeat_feedback_analysis["variants"][REPEAT_FEEDBACK_AYA_DIRECT]["supports_variant"]
        is True
    )


def test_analysis_rejects_candidate_leaking_into_another_family():
    tasks = _tasks()
    runs = _runs({"aya", "mistral"})
    runs[0] = replace(runs[0], candidate_variant=REPEAT_FEEDBACK_MISTRAL_USE)
    with pytest.raises(ValueError, match="exact family/seed/candidate grid"):
        analyze_feedback_adaptation(_design(tasks), runs)


def test_committed_adaptation_design_matches_powered_ledger():
    root = Path(__file__).parents[3]
    design = load_repeat_power_design(
        root / "samples/benchmarks/agentic_loop_feedback_family_adaptation_design.json"
    )
    from llb.bench.agentic.run import load_tasks_file

    tasks = load_tasks_file(root / "samples/benchmarks/agentic_loop_repeat_power_uk.json")
    validate_feedback_adaptation_design(design, tasks)
