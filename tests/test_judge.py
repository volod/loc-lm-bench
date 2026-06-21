from llb.scoring.judge import (
    UA_ANSWER_RELEVANCY_INSTRUCTION,
    UA_FAITHFULNESS_INSTRUCTION,
    extract_scores,
    judge_is_trusted,
    ragas_scorer,
    run_judge,
    to_ragas_samples,
)


def test_gate_helper():
    assert judge_is_trusted(0.6)
    assert not judge_is_trusted(0.59)
    assert not judge_is_trusted(None)


def test_no_judge_configured_is_demoted():
    out = run_judge([], judge_model=None, calibration_rho=0.9)
    assert out.trusted is False and "no judge" in out.reason


def test_uncalibrated_is_demoted():
    out = run_judge([], judge_model="gpt-judge", calibration_rho=None)
    assert out.trusted is False and "not calibrated" in out.reason


def test_below_threshold_is_demoted():
    out = run_judge([], judge_model="gpt-judge", calibration_rho=0.4)
    assert out.trusted is False and "threshold" in out.reason


def test_trusted_routes_to_scorer():
    out = run_judge(
        [{"q": 1}],
        judge_model="gpt-judge",
        calibration_rho=0.8,
        scorer=lambda recs, model: [{"faithfulness": 1.0}],
    )
    assert out.trusted is True
    assert out.scores == [{"faithfulness": 1.0}]


# --- M3.8 Ragas scorer (pure mapping/extraction + injected evaluate) ----------------------


def test_to_ragas_samples_maps_fields():
    samples = to_ragas_samples([{"question": "Q?", "answer": "A", "contexts": ["c1", "c2"]}])
    assert samples == [{"user_input": "Q?", "response": "A", "retrieved_contexts": ["c1", "c2"]}]


def test_extract_scores_tolerates_both_relevancy_keys():
    rows = [
        {"faithfulness": 0.8, "answer_relevancy": 0.7},
        {"faithfulness": 0.5, "response_relevancy": 0.9},  # ragas 0.2 spelling
    ]
    scores = extract_scores(rows)
    assert scores[0] == {"faithfulness": 0.8, "answer_relevancy": 0.7}
    assert scores[1] == {"faithfulness": 0.5, "answer_relevancy": 0.9}


def test_ragas_scorer_uses_injected_evaluate():
    records = [{"question": "Столиця?", "answer": "Київ", "contexts": ["Київ - столиця."]}]

    def fake_evaluate(samples, judge_model):
        assert samples[0]["user_input"] == "Столиця?" and judge_model == "judge-x"
        return [{"faithfulness": 1.0, "answer_relevancy": 0.9}]

    scores = ragas_scorer(records, "judge-x", evaluate_fn=fake_evaluate)
    assert scores == [{"faithfulness": 1.0, "answer_relevancy": 0.9}]


def test_ua_metric_instructions_are_ukrainian():
    assert "ВІРНІСТЬ" in UA_FAITHFULNESS_INSTRUCTION
    assert "РЕЛЕВАНТНІСТЬ" in UA_ANSWER_RELEVANCY_INSTRUCTION
