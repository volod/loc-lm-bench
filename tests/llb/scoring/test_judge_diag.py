"""judge diagnostics judge diagnostics -- classifier, summary, gated wiring, strict-JSON smoke."""

from llb.bench import agentic
from llb.bench.common import run_gated_judge
from llb.judge.experiment import judge_smoke_check
from llb.scoring import judge_diag as jd
from llb.scoring.judge_diag import summarize_judge_diagnostics


def _rec(answer):
    return {"question": "q", "answer": answer, "contexts": ["ctx"]}


# --- classifier ---------------------------------------------------------------------------


def test_classify_record_empty_answer():
    assert jd.classify_record("", {"faithfulness": 0.0, "answer_relevancy": 0.0}) == (
        jd.JUDGE_DIAG_EMPTY_ANSWER
    )
    assert jd.classify_record("   ", {"faithfulness": 0.9, "answer_relevancy": 0.9}) == (
        jd.JUDGE_DIAG_EMPTY_ANSWER
    )


def test_classify_record_precise_reason_wins():
    score = {"faithfulness": 0.0, "answer_relevancy": 0.0}
    assert (
        jd.classify_record("a", score, jd.JUDGE_DIAG_MALFORMED_JSON) == jd.JUDGE_DIAG_MALFORMED_JSON
    )
    assert jd.classify_record("a", score, jd.JUDGE_DIAG_TRANSPORT_ERROR) == (
        jd.JUDGE_DIAG_TRANSPORT_ERROR
    )


def test_classify_record_zero_and_ok():
    assert jd.classify_record("a", {"faithfulness": 0.0, "answer_relevancy": 0.0}) == (
        jd.JUDGE_DIAG_ZERO
    )
    assert (
        jd.classify_record("a", {"faithfulness": 0.1, "answer_relevancy": 0.0}) == jd.JUDGE_DIAG_OK
    )


def test_classify_judge_exception_transport_vs_malformed():
    assert (
        jd.classify_judge_exception(TimeoutError("read timeout")) == jd.JUDGE_DIAG_TRANSPORT_ERROR
    )
    assert jd.classify_judge_exception(ConnectionError("connection refused")) == (
        jd.JUDGE_DIAG_TRANSPORT_ERROR
    )
    assert jd.classify_judge_exception(ValueError("invalid json: expecting value")) == (
        jd.JUDGE_DIAG_MALFORMED_JSON
    )


def test_summarize_counts_reasons():
    records = [_rec("good"), _rec(""), _rec("bad"), _rec("down")]
    scores = [
        {"faithfulness": 0.9, "answer_relevancy": 0.8},
        {"faithfulness": 0.0, "answer_relevancy": 0.0},
        {"faithfulness": 0.0, "answer_relevancy": 0.0},
        {"faithfulness": 0.0, "answer_relevancy": 0.0},
    ]
    reasons = [None, None, jd.JUDGE_DIAG_MALFORMED_JSON, jd.JUDGE_DIAG_TRANSPORT_ERROR]
    diag = summarize_judge_diagnostics(records, scores, reasons)
    assert diag["n"] == 4 and diag["n_ok"] == 1 and diag["n_zero"] == 3
    assert diag["reasons"] == {
        jd.JUDGE_DIAG_EMPTY_ANSWER: 1,
        jd.JUDGE_DIAG_MALFORMED_JSON: 1,
        jd.JUDGE_DIAG_TRANSPORT_ERROR: 1,
    }


def test_summarize_infers_from_scores_without_reasons():
    records = [_rec("a"), _rec("")]
    scores = [
        {"faithfulness": 0.0, "answer_relevancy": 0.0},
        {"faithfulness": 0.0, "answer_relevancy": 0.0},
    ]
    diag = summarize_judge_diagnostics(records, scores)
    assert diag["reasons"] == {jd.JUDGE_DIAG_ZERO: 1, jd.JUDGE_DIAG_EMPTY_ANSWER: 1}


# --- gated-judge wiring -------------------------------------------------------------------


def fixed_scorer(*pairs):
    scores = [{"faithfulness": f, "answer_relevancy": r} for f, r in pairs]
    return lambda records, _model: scores


def test_run_gated_judge_attaches_diagnostics_when_trusted():
    records = [_rec("a"), _rec("")]
    outcome = run_gated_judge(
        records,
        judge_model="judge",
        judge_rho=0.7,
        scorer=fixed_scorer((0.8, 0.6), (0.0, 0.0)),
    )
    assert outcome.trusted is True and outcome.diagnostics is not None
    assert outcome.diagnostics["n_ok"] == 1
    assert outcome.diagnostics["reasons"] == {jd.JUDGE_DIAG_EMPTY_ANSWER: 1}


def test_run_gated_judge_demoted_has_no_diagnostics():
    outcome = run_gated_judge(
        [_rec("a")], judge_model="judge", judge_rho=0.3, scorer=fixed_scorer((0.9, 0.9))
    )
    assert outcome.trusted is False and outcome.diagnostics is None


def test_run_agentic_records_judge_diagnostics(tmp_path):
    tasks = [
        agentic.AgenticTask("a", "p", success=[{"kind": "answer_contains", "value": "x"}]),
        agentic.AgenticTask("b", "p", success=[{"kind": "answer_contains", "value": "x"}]),
    ]
    run = agentic.run_agentic(
        tasks,
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":""}}',  # empty answers
        judge_model="judge",
        judge_rho=0.7,
        judge_scorer=lambda recs, _m: [
            {"faithfulness": 0.0, "answer_relevancy": 0.0} for _ in recs
        ],
        persist=False,
    )
    assert run.judge_diagnostics is not None
    # both candidates produced empty answers -> classified as candidate failures, not judge faults
    assert run.judge_diagnostics["reasons"] == {jd.JUDGE_DIAG_EMPTY_ANSWER: 2}


# --- strict-JSON smoke check --------------------------------------------------------------


def test_judge_smoke_ok():
    result = judge_smoke_check("judge", scorer=fixed_scorer((0.9, 0.8)))
    assert result.ok is True and result.score == {"faithfulness": 0.9, "answer_relevancy": 0.8}


def test_judge_smoke_fails_on_zero():
    result = judge_smoke_check("judge", scorer=fixed_scorer((0.0, 0.0)))
    assert result.ok is False and "zero" in result.reason


def test_judge_smoke_fails_on_malformed_out_of_range():
    result = judge_smoke_check("judge", scorer=fixed_scorer((2.0, 0.5)))
    assert result.ok is False and "well-formed" in result.reason
