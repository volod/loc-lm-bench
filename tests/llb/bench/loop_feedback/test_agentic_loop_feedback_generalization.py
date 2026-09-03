"""Cross-family and cross-seed repeat-feedback aggregation."""

from pathlib import Path

import pytest

from llb.artifacts.runs.bundle import read_study_analysis
from llb.bench.agentic.loop_policy import REPEAT_FEEDBACK_BILINGUAL
from llb.bench.agentic.model import AgenticTask
from llb.bench.context_policy.run import task_set_digest
from llb.bench.loop_feedback.generalization import (
    FeedbackSeedRun,
    analyze_feedback_generalization,
    validate_feedback_generalization_design,
)
from llb.bench.loop_feedback.generalization_report import (
    format_feedback_generalization_table,
    persist_feedback_generalization,
)


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


def _design(tasks: list[AgenticTask]) -> dict[str, object]:
    families = ["aya", "mistral", "qwen", "gemma"]
    return {
        "schema_version": 1,
        "study_id": "generalization-test",
        "study_kind": "repeat_feedback_generalization",
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
        "reference_model_families": ["gemma", "qwen"],
        "minimum_additional_model_families": 2,
        "roster": [
            {"model_family": family, "model": f"{family}-model", "backend": "ollama"}
            for family in families
        ],
        "run_seeds": [13, 29],
        "sampling": {"temperature": 0.2},
        "cross_family_adoption_rule": {
            "minimum_supported_seeds_per_family": 2,
            "minimum_supported_families": 3,
            "minimum_supported_fraction": 0.75,
            "require_additional_family_support": True,
        },
        "repeat_feedback_variants": ["current", "bilingual"],
        "fixed_policy": {
            "max_steps": 6,
            "malformed_call": "answer",
            "repeated_call": ["allow", "noop"],
        },
    }


def _analysis(family: str, seed: int, supports: bool) -> dict[str, object]:
    completion_delta = 0.5 if supports else 0.0
    delta = {"mean": completion_delta, "lo": completion_delta, "hi": completion_delta}
    cost_delta = {"mean": -10.0, "lo": -12.0, "hi": -8.0}
    return {
        "study_id": "generalization-test",
        "model_family": family,
        "run_seed": seed,
        "coverage_passed": True,
        "baseline": {"activation_rate": 0.75, "activation_passed": True},
        "variants": {
            REPEAT_FEEDBACK_BILINGUAL: {
                "activation_passed": True,
                "activation_rate": 0.75,
                "supports_variant": supports,
                "completion_rate": 0.75 if supports else 0.25,
                "redirect": {"response_rate": 0.75 if supports else 0.25},
                "completion": {"paired": {"delta": delta}},
                "cost": {
                    "total_model_input_tokens": {"paired_delta": cost_delta},
                    "elapsed_s": {"paired_delta": cost_delta},
                },
            }
        },
    }


def _runs(supported_families: set[str]) -> list[FeedbackSeedRun]:
    return [
        FeedbackSeedRun(
            family,
            f"{family}-model",
            seed,
            _analysis(family, seed, family in supported_families),
            {"bilingual": f"/runs/{family}/{seed}/manifest.json"},
        )
        for family in ["aya", "mistral", "qwen", "gemma"]
        for seed in [13, 29]
    ]


def test_generalization_design_requires_independent_families_and_real_sampling():
    tasks = _tasks()
    design = _design(tasks)
    validate_feedback_generalization_design(design, tasks)
    design["sampling"] = {"temperature": 0.0}
    with pytest.raises(ValueError, match="temperature"):
        validate_feedback_generalization_design(design, tasks)


def test_generalization_design_requires_boolean_additional_family_gate():
    tasks = _tasks()
    design = _design(tasks)
    design["cross_family_adoption_rule"]["require_additional_family_support"] = "true"
    with pytest.raises(ValueError, match="must be boolean"):
        validate_feedback_generalization_design(design, tasks)


def test_generalization_aggregates_stable_seed_support_and_persists(tmp_path: Path):
    tasks = _tasks()
    design = _design(tasks)
    analysis = analyze_feedback_generalization(
        design,
        _runs({"aya", "mistral", "qwen"}),
    )
    assert analysis["coverage_and_activation_passed"] is True
    assert analysis["stable_supported_families"] == ["aya", "mistral", "qwen"]
    assert analysis["supports_global_feedback_default"] is True
    assert analysis["recommended_global_feedback_variant"] == "bilingual"
    assert analysis["families"]["gemma"]["routed_feedback_variant"] == "current"
    table = format_feedback_generalization_table(analysis)
    assert "family routing" in table and "mistral" in table

    paths = persist_feedback_generalization(
        design,
        analysis,
        data_dir=tmp_path,
        task_digest=task_set_digest(tasks),
        table=table,
        mirror=lambda *_args: None,
    )
    run_dir = Path(paths["manifest"]).parent
    persisted = read_study_analysis(run_dir / "generalization-analysis.json")
    assert persisted["supported_family_fraction"] == 0.75
    assert persisted["seed_rows"][0]["coverage_passed"] is True
    assert persisted["seed_rows"][0]["variant_activation_rate"] == 0.75


def test_generalization_rejects_missing_seed_cell():
    tasks = _tasks()
    design = _design(tasks)
    runs = _runs({"aya", "mistral", "qwen"})[:-1]
    with pytest.raises(ValueError, match="exact family/seed grid"):
        analyze_feedback_generalization(design, runs)


def test_generalization_rejects_mismatched_analysis_coordinate():
    tasks = _tasks()
    design = _design(tasks)
    runs = _runs({"qwen"})
    runs[0].analysis["run_seed"] = 999
    with pytest.raises(ValueError, match="metadata"):
        analyze_feedback_generalization(design, runs)
