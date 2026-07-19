"""Frozen bilingual bundle, paired report, and CLI registration tests."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bilingual_cutoff_helpers import accept_all, loaded_events, translation
from llb.bench.knowledge_cutoff.paired import load_reviewed_lanes, run_bilingual_cutoff
from llb.bench.knowledge_cutoff.paired_report import paired_statistics
from llb.bench.knowledge_cutoff.translation_workflow import draft_translation_bundle
from llb.bench.knowledge_cutoff.translation_review import (
    REVIEW_SUMMARY_FILENAME,
    freeze_reviewed_bundle,
    review_bundle_status,
)
from llb.cli import app


def test_frozen_bundle_runs_aligned_paired_report(tmp_path):
    bundle = tmp_path / "translation"
    draft_translation_bundle(
        loaded_events(), complete=translation, out_dir=bundle, translator="local"
    )
    accept_all(bundle)
    assert review_bundle_status(bundle)["ready_to_freeze"] is True
    summary = freeze_reviewed_bundle(bundle, reviewer="reviewer-1")
    assert summary["accepted_rows"] == 4
    english, ukrainian, review = load_reviewed_lanes(bundle)
    assert [item.id for item in english.events] == [item.id for item in ukrainian.events]
    assert "Яке" in ukrainian.events[0].mcq_question
    assert review["resolved_revision"] == "a" * 40

    def complete(prompt: str) -> str:
        marker = "correct marker" if "correct marker" in prompt else "правильна позначка"
        return next(line[0] for line in prompt.splitlines() if marker in line)

    result = run_bilingual_cutoff(
        bundle,
        model="local-test",
        backend="ollama",
        complete=complete,
        data_dir=tmp_path / "runs",
        optuna_trials=5,
    )
    assert result.paired["accuracy_delta"] == 0.0
    assert result.paths is not None
    report_dir = Path(result.paths["manifest"]).parent
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert report["paired"]["bootstrap"]["samples"] == 2000
    assert "Monthly Language Deltas" in (report_dir / "report.md").read_text(encoding="utf-8")
    assert (bundle / REVIEW_SUMMARY_FILENAME).is_file()


def test_paired_statistics_detects_choice_mapping_drift():
    english = [
        {
            "item_id": "e1",
            "month": "2025-01",
            "counts_for_curve": True,
            "choice_order": ["A", "B", "C", "D"],
            "expected": "B",
            "objective_score": 1.0,
        }
    ]
    ukrainian = [{**english[0], "choice_order": ["B", "A", "C", "D"]}]
    with pytest.raises(ValueError, match="different source-choice mappings"):
        paired_statistics(english, ukrainian, seed=42)


@pytest.mark.parametrize(
    "command",
    [
        "knowledge-cutoff-ua-draft",
        "knowledge-cutoff-ua-review",
        "knowledge-cutoff-ua-revise",
        "knowledge-cutoff-ua-validate",
        "knowledge-cutoff-ua-freeze",
        "bench-knowledge-cutoff-bilingual",
    ],
)
def test_bilingual_cli_commands_are_registered(command):
    result = CliRunner().invoke(app, [command, "--help"])
    assert result.exit_code == 0
