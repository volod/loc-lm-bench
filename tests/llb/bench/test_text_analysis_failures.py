"""Tests for text analysis failures."""

import json
from llb.bench.text_analysis.run import run_text_analysis
from test_bench_text_analysis import ZERO_SIM, _write_bundle


def test_run_text_analysis_malformed_output(tmp_path):
    bundle = _write_bundle(tmp_path / "b")
    run = run_text_analysis(
        bundle,
        model="m",
        backend="ollama",
        complete=lambda _: "not json at all",
        similarity=ZERO_SIM,
        persist=False,
    )
    assert run.rows[0]["status"] == "malformed"
    assert run.result.objective_score == 0.0
    assert run.result.reliability == 0.0


def test_run_text_analysis_hallucination_penalizes_precision(tmp_path):
    bundle = _write_bundle(tmp_path / "b")
    # recover the entities but also hallucinate an extra one -> precision < 1
    run = run_text_analysis(
        bundle,
        model="m",
        backend="ollama",
        complete=lambda _: json.dumps(
            {"entity": ["Київ", "Львів", "Марс"], "topic": ["економіка"], "insight": []}
        ),
        similarity=ZERO_SIM,
        persist=False,
    )
    f1 = json.loads(run.rows[0]["subtask_f1_json"])
    assert f1["entity"] < 1.0  # the hallucinated "Марс" lowered entity precision
