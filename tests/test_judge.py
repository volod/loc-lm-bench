from llb.scoring.judge import judge_is_trusted, run_judge


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
