"""Tests for board ranking."""

import json
from llb.board.io import read_case_series
from llb.board.runs import (
    best_per_model,
    load_run_records,
    record_from_manifest,
)
from test_board import _write_run


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
