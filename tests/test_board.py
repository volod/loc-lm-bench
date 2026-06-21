"""Board data loaders (M3.7): manifest/scores -> ModelResults + best-per-model (pure)."""

import json

from llb.board.data import (
    best_per_model,
    config_summary,
    load_run_records,
    read_case_objectives,
    record_from_manifest,
)


def _write_run(root, name, model, objective, cases, backend="ollama", strategy="markdown"):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": name,
        "run_name": name,
        "created_at": "2026-06-21T00:00:00Z",
        "config": {"model": model, "backend": backend, "strategy": strategy, "top_k": 6},
        "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 50.0},
        "telemetry": {"peak_vram_mb": 5500},
        "n_cases": len(cases),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "scores.jsonl").write_text(
        "\n".join(json.dumps({"objective_score": c}) for c in cases), encoding="utf-8"
    )
    return run_dir


def test_read_case_objectives_jsonl(tmp_path):
    run = _write_run(tmp_path, "20260101T000000Z-aaa", "m:1", 0.7, [1.0, 0.0, 1.0])
    assert read_case_objectives(run) == [1.0, 0.0, 1.0]


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
