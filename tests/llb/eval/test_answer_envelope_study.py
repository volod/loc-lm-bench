"""typed-rag-answer-envelope -- the roster conformance study over recorded envelope bundles.

The study is pure over finished bundles, so the whole thing runs on bundles the fake-driven runner
wrote into tmp_path: two models, one item set, no backend and no GPU.
"""

import json

import pytest

from llb.eval.answer_envelope.study import analyze, render
from tests.llb.eval.test_answer_envelope_scoring import _valid, run_envelope_eval

# Every test here builds its bundles through the REAL runner, which compiles the LangGraph app
# (`llb.eval.graph.build_rag_graph`) and therefore needs the `[eval]` extra the base [dev] install
# of GitHub CI lacks. The study code they exercise is pure; only the bundle fixture is heavy.
pytestmark = pytest.mark.heavy_env

MALFORMED = "просто проза"
INVALID = '{"answer": "Так"}'


def _bundle(tmp_path, name, responses, model):
    cfg, result, _ = run_envelope_eval(tmp_path / name, responses, model=model)
    return cfg.run_dir(result["run_timestamp"])


def _roster(tmp_path):
    """Two models over the same two items: one fully conformant, one repaired into conformance."""
    clean = _bundle(tmp_path, "clean", [_valid(), _valid()], model="model-a")
    repaired = _bundle(tmp_path, "repaired", [INVALID, _valid(), MALFORMED, MALFORMED], "model-b")
    return clean, repaired


def test_the_study_separates_format_from_reasoning(tmp_path):
    clean, repaired = _roster(tmp_path)
    report = analyze([clean, repaired])
    a, b = report["models"]["model-a"], report["models"]["model-b"]
    assert report["n"] == 2
    assert (a["conformance"], a["repair_rate"]) == (1.0, 0.0)
    # model-b failed both first attempts; the repair rescued one of them.
    assert (b["conformance"], b["repair_rate"]) == (0.5, 1.0)
    assert b["first_attempt_conformance"] == 0.0
    assert b["repair_gain"] == 0.5
    assert b["malformed_rate"] == 0.5
    # correctness stands beside conformance as its own column, never folded into it
    assert "objective_score" in a and "contains" in a
    assert report["conformance_order"] == ["model-a", "model-b"]


def test_the_rendered_report_states_both_readings(tmp_path):
    clean, repaired = _roster(tmp_path)
    text = render(analyze([clean, repaired]))
    assert "conformance" in text and "first attempt" in text
    assert "FORMATTING gain" in text
    assert "`model-a`" in text and "`model-b`" in text


def test_a_truncated_completion_is_flagged_rather_than_blamed_on_the_model(tmp_path):
    # A completion cut off at the token cap is not JSON either; counting it as "cannot emit the
    # shape" would make the budget look like a model property.
    clean, repaired = _roster(tmp_path)
    scores = repaired / "scores.jsonl"
    rows = [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["completion_tokens"] = 512  # the shipped max_tokens default this run used
    scores.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    report = analyze([clean, repaired])
    assert report["models"]["model-b"]["truncation_suspect_rate"] == 1.0
    assert report["models"]["model-a"]["truncation_suspect_rate"] == 0.0  # nothing failed at all


def test_one_bundle_is_not_a_roster(tmp_path):
    clean, _ = _roster(tmp_path)
    with pytest.raises(ValueError, match="at least two"):
        analyze([clean])


def test_the_same_model_twice_is_not_a_roster(tmp_path):
    first = _bundle(tmp_path, "first", [_valid(), _valid()], model="model-a")
    second = _bundle(tmp_path, "second", [_valid(), _valid()], model="model-a")
    with pytest.raises(ValueError, match="duplicate"):
        analyze([first, second])


def test_a_free_text_bundle_is_refused_before_any_number_is_read(tmp_path):
    clean, _ = _roster(tmp_path)
    manifest_path = clean / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["config"]["answer_format"] = "free_text"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not the envelope lane"):
        analyze([clean, clean])


def test_bundles_over_different_item_sets_are_not_comparable(tmp_path):
    clean, repaired = _roster(tmp_path)
    scores = repaired / "scores.jsonl"
    rows = [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines()]
    rows[0]["item_id"] = "uk-99"
    scores.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different item sets"):
        analyze([clean, repaired])
