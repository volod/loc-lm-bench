"""Board data loaders (M3.7): manifest/scores -> ModelResults + best-per-model (pure)."""

import json

from llb.board.data import (
    best_per_model,
    config_summary,
    load_run_records,
    load_screen_reports,
    read_case_objectives,
    read_case_series,
    read_case_splits,
    record_from_manifest,
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


# --- M3.7 board completion: judge/semantic series, policy best-pick, Tier-1 separation -----


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
