"""Board data loaders: manifest/scores -> ModelResults + best-per-model."""

import json

from llb.board.categories import (
    load_category_composite,
    load_category_records,
    load_category_run_records,
)
from llb.board.io import read_case_objectives, read_case_series, read_case_splits
from llb.board.prompt_systems import (
    load_rag_prompt_system_records,
    rag_prompt_system_comparison,
)
from llb.board.runs import (
    best_per_model,
    config_summary,
    load_run_records,
    load_screen_reports,
    record_from_manifest,
)
from llb.scoring.aggregate import (
    TIER_AGENTIC,
    TIER_SECURITY,
    TIER_STRUCTURED,
    TIER_SUMMARIZATION,
    TIER_TEXT_ANALYSIS,
    TIER_TOOLING,
)


def _write_run(
    root,
    name,
    model,
    objective,
    cases,
    backend="ollama",
    strategy="markdown",
    split="final",
    judge=None,
    semantic=None,
    prompt_system=None,
):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": name,
        "run_name": name,
        "split": split,
        "created_at": "2026-06-21T00:00:00Z",
        "config": {"model": model, "backend": backend, "strategy": strategy, "top_k": 6},
        "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 50.0},
        "telemetry": {"peak_vram_mb": 5500},
        "n_cases": len(cases),
    }
    if prompt_system is not None:
        manifest["config"]["prompt_system"] = prompt_system
        manifest["prompt_system_provenance"] = {
            "prompt_system_id": prompt_system,
            "corpus_digest": "corpus",
            "mapping_digest": "mapping",
            "template_revision": "template",
            "tokenizer": "char-ratio",
            "context_window": 4096,
            "prompt_budget_tokens": 3000,
        }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for i, c in enumerate(cases):
        row = {"objective_score": c, "split": split}
        if judge is not None:
            row["judge_score"] = judge[i]
        if semantic is not None:
            row["semantic"] = semantic[i]
        rows.append(json.dumps(row))
    (run_dir / "scores.jsonl").write_text("\n".join(rows), encoding="utf-8")
    return run_dir


def test_read_case_objectives_jsonl(tmp_path):
    run = _write_run(tmp_path, "20260101T000000Z-aaa", "m:1", 0.7, [1.0, 0.0, 1.0])
    assert read_case_objectives(run) == [1.0, 0.0, 1.0]
    assert read_case_splits(run) == {"final"}


def test_record_from_manifest_builds_model_result(tmp_path):
    run = _write_run(tmp_path, "r1", "m:1", 0.7, [1.0, 0.0])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    rec = record_from_manifest(manifest, run)
    assert rec is not None
    assert rec.result.model == "m:1" and rec.result.objective_score == 0.7
    assert rec.result.peak_vram_mb == 5500 and rec.result.case_objectives == [1.0, 0.0]


def test_record_from_manifest_none_without_model(tmp_path):
    assert record_from_manifest({"config": {}}, tmp_path) is None


def test_load_run_records_skips_staging_dirs(tmp_path):
    _write_run(tmp_path, "20260101T000001Z-aaa", "m:1", 0.7, [1.0])
    staging = tmp_path / ".20260101T000002Z-bbb.tmp"
    staging.mkdir()
    (staging / "manifest.json").write_text(json.dumps({"config": {"model": "x"}}), encoding="utf-8")
    records = load_run_records(tmp_path)
    assert [r.result.model for r in records] == ["m:1"]  # staging .tmp ignored


def test_load_run_records_excludes_tuning_and_calibration_runs(tmp_path):
    _write_run(tmp_path, "final", "m:1", 0.6, [0.6], split="final")
    _write_run(tmp_path, "tuning", "m:1", 0.99, [0.99], split="tuning")
    _write_run(tmp_path, "calibration", "m:1", 1.0, [1.0], split="calibration")
    records = load_run_records(tmp_path)
    assert [(r.result.objective_score, r.split) for r in records] == [(0.6, "final")]


def test_record_from_legacy_manifest_infers_non_final_split_from_scores(tmp_path):
    run = _write_run(tmp_path, "legacy", "m:1", 0.99, [0.99], split="tuning")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("split")
    assert record_from_manifest(manifest, run) is None


def test_record_from_legacy_manifest_accepts_final_split_from_scores(tmp_path):
    run = _write_run(tmp_path, "legacy-final", "m:1", 0.7, [0.7], split="final")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("split")
    record = record_from_manifest(manifest, run)
    assert record is not None and record.split == "final"


def test_best_per_model_keeps_highest_objective(tmp_path):
    _write_run(tmp_path, "r1", "m:1", 0.6, [1.0])
    _write_run(tmp_path, "r2", "m:1", 0.8, [1.0])  # better config for the same model
    _write_run(tmp_path, "r3", "m:2", 0.5, [1.0])
    best = best_per_model(load_run_records(tmp_path))
    by_model = {r.result.model: r.result.objective_score for r in best}
    assert by_model == {"m:1": 0.8, "m:2": 0.5}


def test_config_summary_selects_known_keys():
    summary = config_summary({"strategy": "markdown", "top_k": 6, "model": "m", "junk": 1})
    assert summary["strategy"] == "markdown" and summary["top_k"] == 6
    assert "model" not in summary and "junk" not in summary


def test_rag_prompt_system_records_and_comparison_use_final_run_eval_bundles(tmp_path):
    run_root = tmp_path / "run-eval"
    _write_run(run_root, "ps1", "m:1", 0.25, [0.0, 0.5], prompt_system="ps1")
    _write_run(run_root, "ps2-weak", "m:1", 0.1, [0.1], prompt_system="ps2")
    _write_run(run_root, "ps2-strong", "m:1", 0.9, [1.0, 0.8], prompt_system="ps2")
    _write_run(run_root, "tuning", "m:1", 1.0, [1.0], split="tuning", prompt_system="leak")

    records = load_rag_prompt_system_records(tmp_path)
    assert {(r.model, r.prompt_system, r.result.objective_score) for r in records} == {
        ("m:1", "ps1", 0.25),
        ("m:1", "ps2", 0.9),
    }

    rows, table, ids = rag_prompt_system_comparison(tmp_path, "m:1")
    assert ids == ["ps1", "ps2"]
    assert rows[0]["model"] == "ps2"
    assert "policy:" in table


# --- Board completion: judge/semantic series, policy best-pick, Tier-1 separation -----------


def test_record_loads_semantic_and_judge_series(tmp_path):
    run = _write_run(tmp_path, "r1", "m:1", 0.7, [1.0, 0.0], judge=[0.8, 0.6], semantic=[0.9, 0.7])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    rec = record_from_manifest(manifest, run)
    assert rec is not None
    assert rec.result.case_judge == [0.8, 0.6] and rec.result.case_semantic == [0.9, 0.7]
    assert rec.result.judge_score == 0.7 and rec.result.semantic_score == 0.8  # means
    assert read_case_series(run, "judge_score") == [0.8, 0.6]


def test_best_per_model_uses_ranking_policy_not_objective(tmp_path):
    # config A: higher objective, low judge; config B: lower objective, high judge.
    _write_run(tmp_path, "a", "m:1", 0.8, [0.8, 0.8], judge=[0.1, 0.1])
    _write_run(tmp_path, "b", "m:1", 0.6, [0.6, 0.6], judge=[1.0, 1.0])
    records = load_run_records(tmp_path)
    # objective-only would keep A; the trusted-judge policy (blend 0.5) keeps B (0.6*.5+1*.5=0.8).
    best = best_per_model(records, judge_trusted=True, weight_judge=0.5)
    assert len(best) == 1 and best[0].result.objective_score == 0.6


def test_load_screen_reports_separates_tier1(tmp_path):
    screens = tmp_path / "screen"
    screens.mkdir()
    (screens / "m.json").write_text(
        json.dumps({"model": "m", "track": "logprob", "results": [{"task": "t", "score": 0.5}]}),
        encoding="utf-8",
    )
    (screens / "junk.json").write_text(json.dumps({"not": "a report"}), encoding="utf-8")
    reports = load_screen_reports(screens)
    assert len(reports) == 1 and reports[0]["track"] == "logprob"


# --- Category boards (each its own Tier, never cross-ranked) --------------------------------


def _write_category_run(
    data_dir,
    method,
    tier,
    model,
    objective,
    n_cases=4,
    *,
    data_verified=False,
    verification_ref=None,
    scores=None,
):
    safe_model = model.replace(":", "_")
    run_dir = data_dir / method / f"20260101T000000Z-{method}-{safe_model}-{objective}"
    run_dir.mkdir(parents=True)
    scores = scores if scores is not None else []
    if data_verified and verification_ref is None:
        verification_ref = _write_verification_ref(data_dir)
    manifest = {
        "run_id": f"{method}-{model}",
        "split": "final",
        "created_at": "2026-06-21T00:00:00Z",
        "config": {
            "model": model,
            "backend": "ollama",
            "tier": tier,
            "category": method,
            "data_verified": data_verified,
            "verification_ref": str(verification_ref) if verification_ref else None,
        },
        "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 0.0},
        "n_cases": n_cases,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [json.dumps({"objective_score": score}) for score in scores]
    (run_dir / "scores.jsonl").write_text("\n".join(rows), encoding="utf-8")
    return run_dir


def _write_verification_ref(root):
    path = root / "verify_sample.csv"
    path.write_text("item_id,stratum,decision\nsample,s,accept\n", encoding="utf-8")
    return path


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


def _write_full_composite_model(tmp_path, model, objectives, *, data_verified=True):
    tiers = [
        ("text-analysis", TIER_TEXT_ANALYSIS),
        ("summarization", TIER_SUMMARIZATION),
        ("structured", TIER_STRUCTURED),
        ("security", TIER_SECURITY),
        ("agentic", TIER_AGENTIC),
        ("tooling", TIER_TOOLING),
    ]
    for method, tier in tiers:
        objective = objectives[tier]
        _write_category_run(
            tmp_path,
            method,
            tier,
            model,
            objective,
            n_cases=2,
            data_verified=data_verified,
            scores=[objective, objective],
        )


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
