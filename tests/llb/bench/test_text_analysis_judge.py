"""Tests for text analysis judge."""

import json
from llb.bench.text_analysis.prompts import parse_predictions
from llb.bench.text_analysis.run import run_text_analysis
from llb.scoring import text_analysis_labels as ta
from test_bench_text_analysis import ZERO_SIM, _write_bundle, _write_long_doc_bundle, fake_judge


def test_parse_predictions_coerces_scalar_and_missing():
    preds = parse_predictions(json.dumps({"entity": "Київ"}), [ta.ENTITY, ta.TOPIC])
    assert preds[ta.ENTITY] == ["Київ"]  # scalar coerced to a one-item list
    assert preds[ta.TOPIC] == []  # missing kind -> empty


def test_gated_judge_scores_narrative_insight_alongside_objective(tmp_path):
    bundle = _write_bundle(tmp_path / "b")  # plants entity/topic objective + insight judged

    run = run_text_analysis(
        bundle,
        model="m",
        backend="ollama",
        complete=lambda _: json.dumps(
            {"entity": ["Київ", "Львів"], "topic": ["економіка"], "insight": ["ринок зростає"]}
        ),
        similarity=ZERO_SIM,
        judge_model="judge",
        judge_rho=0.7,  # trusted
        judge_scorer=fake_judge(0.8, 1.0),
        persist=False,
    )
    assert run.judge_trusted is True
    assert run.judged_quality == 0.9  # (0.8 + 1.0)/2
    assert run.result.objective_score == 1.0  # objective headline unchanged by the judge
    assert run.rows[0]["judged_quality"] == 0.9


def test_gated_judge_demoted_below_threshold(tmp_path):
    bundle = _write_bundle(tmp_path / "b")
    run = run_text_analysis(
        bundle,
        model="m",
        backend="ollama",
        complete=lambda _: json.dumps({"entity": ["Київ"], "topic": [], "insight": ["x"]}),
        similarity=ZERO_SIM,
        judge_model="judge",
        judge_rho=0.4,  # below the 0.6 gate
        judge_scorer=fake_judge(1.0, 1.0),
        persist=False,
    )
    assert run.judge_trusted is False and run.judged_quality is None


def test_long_doc_driven_through_map_reduce(tmp_path):
    bundle = _write_long_doc_bundle(tmp_path / "b")
    calls = []

    def complete(prompt):
        calls.append(prompt)
        # the reduce/map prompts ask the comprehension question -> answer with the fact
        return "Бюджет зріс на 15 відсотків."

    run = run_text_analysis(
        bundle,
        model="m",
        backend="ollama",
        complete=complete,
        similarity=ZERO_SIM,
        judge_model="judge",
        judge_rho=0.7,
        judge_scorer=fake_judge(1.0, 1.0),
        persist=False,
    )
    # the long doc was split into multiple segments -> more than one map call
    assert len(calls) > 1
    assert run.rows[0].get("long_doc_answer")  # the map-reduce answer recorded
    assert run.judged_quality == 1.0  # the long_doc answer judged
