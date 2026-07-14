"""Tests for board categories."""

from llb.board.categories import (
    load_category_composite,
    load_category_records,
    load_category_run_records,
)
from llb.scoring.aggregate import (
    TIER_AGENTIC,
    TIER_SECURITY,
    TIER_STRUCTURED,
    TIER_SUMMARIZATION,
    TIER_TEXT_ANALYSIS,
    TIER_TOOLING,
)
from test_board import _write_category_run, _write_full_composite_model


def test_load_category_records_groups_by_tier(tmp_path):
    _write_category_run(tmp_path, "security", TIER_SECURITY, "m:1", 0.4)
    _write_category_run(tmp_path, "tooling", TIER_TOOLING, "m:1", 0.75)
    by_tier = load_category_records(tmp_path)
    assert set(by_tier) == {TIER_SECURITY, TIER_TOOLING}  # each its own tier, not merged
    assert by_tier[TIER_SECURITY][0].objective_score == 0.4
    assert by_tier[TIER_TOOLING][0].tier == TIER_TOOLING


def test_load_category_records_best_per_model_within_tier(tmp_path):
    _write_category_run(tmp_path, "security", TIER_SECURITY, "m:1", 0.4)
    _write_category_run(tmp_path, "security", TIER_SECURITY, "m:1", 0.6)  # better run, same model
    by_tier = load_category_records(tmp_path)
    sec = by_tier[TIER_SECURITY]
    assert len(sec) == 1 and sec[0].objective_score == 0.6  # kept the best


def test_load_category_records_empty(tmp_path):
    assert load_category_records(tmp_path) == {}


def test_load_category_run_records_preserves_verification_metadata(tmp_path):
    _write_category_run(
        tmp_path,
        "security",
        TIER_SECURITY,
        "m:1",
        0.5,
        data_verified=True,
        scores=[1.0, 0.0],
    )
    by_tier = load_category_run_records(tmp_path)
    rec = by_tier[TIER_SECURITY][0]
    assert rec.data_verified is True
    assert rec.verification_ref is not None and rec.verification_ref.endswith("verify_sample.csv")
    assert rec.verification_error is None
    assert rec.result.case_objectives == [1.0, 0.0]


def test_load_category_composite_requires_verified_data(tmp_path):
    objectives = {
        TIER_TEXT_ANALYSIS: 1.0,
        TIER_SUMMARIZATION: 1.0,
        TIER_STRUCTURED: 1.0,
        TIER_SECURITY: 1.0,
        TIER_AGENTIC: 1.0,
        TIER_TOOLING: 1.0,
    }
    _write_full_composite_model(tmp_path, "m:1", objectives, data_verified=False)
    rows, issues = load_category_composite(tmp_path)
    assert rows == []
    assert {issue.reason for issue in issues} == {"category data is not verified"}


def test_load_category_composite_scores_complete_verified_models(tmp_path):
    weak = {
        TIER_TEXT_ANALYSIS: 0.2,
        TIER_SUMMARIZATION: 0.2,
        TIER_STRUCTURED: 0.2,
        TIER_SECURITY: 0.2,
        TIER_AGENTIC: 0.2,
        TIER_TOOLING: 0.2,
    }
    strong = {
        TIER_TEXT_ANALYSIS: 1.0,
        TIER_SUMMARIZATION: 1.0,
        TIER_STRUCTURED: 1.0,
        TIER_SECURITY: 1.0,
        TIER_AGENTIC: 1.0,
        TIER_TOOLING: 1.0,
    }
    _write_full_composite_model(tmp_path, "weak", weak)
    _write_full_composite_model(tmp_path, "strong", strong)
    rows, issues = load_category_composite(tmp_path)
    assert issues == []
    assert [row["model"] for row in rows] == ["strong", "weak"]
    assert rows[0]["score"] == 1.0
    assert rows[1]["score"] == 0.2
    assert "score_ci" in rows[0]


def test_load_category_composite_reports_missing_tier(tmp_path):
    _write_category_run(
        tmp_path,
        "security",
        TIER_SECURITY,
        "m:1",
        1.0,
        data_verified=True,
        scores=[1.0, 1.0],
    )
    rows, issues = load_category_composite(tmp_path)
    assert rows == []
    assert any(issue.reason == "missing required tier" for issue in issues)


def test_load_category_composite_rejects_invalid_verification_ref(tmp_path):
    objectives = {
        TIER_TEXT_ANALYSIS: 1.0,
        TIER_SUMMARIZATION: 1.0,
        TIER_STRUCTURED: 1.0,
        TIER_SECURITY: 1.0,
        TIER_AGENTIC: 1.0,
        TIER_TOOLING: 1.0,
    }
    tiers = [
        ("text-analysis", TIER_TEXT_ANALYSIS),
        ("summarization", TIER_SUMMARIZATION),
        ("structured", TIER_STRUCTURED),
        ("security", TIER_SECURITY),
        ("agentic", TIER_AGENTIC),
        ("tooling", TIER_TOOLING),
    ]
    bad_ref = tmp_path / "missing.csv"
    for method, tier in tiers:
        _write_category_run(
            tmp_path,
            method,
            tier,
            "m:1",
            objectives[tier],
            n_cases=2,
            data_verified=True,
            verification_ref=bad_ref,
            scores=[1.0, 1.0],
        )
    rows, issues = load_category_composite(tmp_path)
    assert rows == []
    assert any("verification ref invalid" in issue.reason for issue in issues)
