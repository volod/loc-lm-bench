"""Resolution effect report verdicts and objective manifest loading."""

import json

from llb.conflicts.resolution_effect import objective_from_manifest, render_effect, write_effect
from llb.rag.refresh.drift import RetrievalDrift


def test_effect_adopts_only_when_measured_axes_do_not_regress(tmp_path):
    drift = RetrievalDrift(3, 10, 0.5, 0.4, 0.6, 0.4, 0.05)
    report = render_effect(drift, before_objective=0.7, after_objective=0.7)
    assert "Verdict: ADOPT" in report
    regressed = render_effect(drift, before_objective=0.7, after_objective=0.6)
    assert "Verdict: REVERT" in regressed


def test_effect_never_adopts_with_pending_objective_or_review():
    drift = RetrievalDrift(3, 10, 0.5, 0.4, 0.5, 0.4, 0.05)
    assert "MEASUREMENT REQUIRED" in render_effect(drift)
    reviewed = render_effect(drift, before_objective=0.7, after_objective=0.7, unresolved_reviews=2)
    assert "Verdict: REVERT" in reviewed


def test_objective_reads_run_manifest(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"metrics": {"objective_score": 0.625}}), encoding="utf-8"
    )
    assert objective_from_manifest(run) == 0.625


def test_cached_retrieval_measurement_is_scoped_to_the_overlay(tmp_path):
    path = tmp_path / "effect.md"
    drift = RetrievalDrift(3, 10, 0.5, 0.4, 0.6, 0.5, 0.05)
    write_effect(path, drift, effect_key="overlay-a")
    write_effect(
        path,
        None,
        effect_key="overlay-a",
        before_objective=0.7,
        after_objective=0.7,
    )
    assert "0.5000" in path.read_text(encoding="utf-8")
    write_effect(path, None, effect_key="overlay-b")
    assert "n/a" in path.read_text(encoding="utf-8")
