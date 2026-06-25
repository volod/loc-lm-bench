"""M5.4 summarization -- reference coverage via injected cosine."""

from llb.bench import summarization as bench_sm
from llb.scoring.aggregate import TIER_SUMMARIZATION


def make_similarity(table):
    def similarity(a, b):
        return table.get((a, b), table.get((b, a), 1.0 if a == b else 0.0))

    return similarity


def test_split_sentences():
    assert bench_sm.split_sentences("Перше. Друге! Третє?") == ["Перше", "Друге", "Третє"]
    assert bench_sm.split_sentences("  ") == []


def test_reference_coverage_partial():
    sim = make_similarity({("друге речення", "перше речення"): 0.4})
    cov = bench_sm.reference_coverage("перше речення. друге речення", "перше речення", sim)
    assert round(cov, 4) == round((1.0 + 0.4) / 2, 4)


def test_reference_coverage_empty_sides():
    sim = make_similarity({})
    assert bench_sm.reference_coverage("", "щось", sim) == 0.0
    assert bench_sm.reference_coverage("щось", "", sim) == 0.0


def test_run_summarization_persists(tmp_path):
    cases = [
        bench_sm.SummarizationCase("a", "довгий документ", "ключовий факт"),
        bench_sm.SummarizationCase("b", "інший документ", "інший факт"),
    ]
    # identity similarity: the model echoes the reference exactly -> coverage 1.0
    run = bench_sm.run_summarization(
        cases,
        model="m",
        backend="ollama",
        complete=lambda prompt: "ключовий факт" if "довгий" in prompt else "інший факт",
        similarity=make_similarity({}),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    assert run.result.tier == TIER_SUMMARIZATION
    assert run.result.objective_score == 1.0
    assert run.coverage_ci is not None
    assert run.paths is not None and "summarization" in run.paths["manifest"]


def test_run_summarization_empty_output_is_unreliable():
    cases = [bench_sm.SummarizationCase("a", "doc", "ref")]
    run = bench_sm.run_summarization(
        cases,
        model="m",
        backend="ollama",
        complete=lambda _: "",
        similarity=make_similarity({}),
        persist=False,
    )
    assert run.rows[0]["status"] == "empty"
    assert run.result.reliability == 0.0


def test_load_committed_summarization_cases():
    cases = bench_sm.load_cases_file("samples/summarization_cases_uk.json")
    assert len(cases) == 3 and all(c.reference for c in cases)


# --- opt-in gated-judge faithfulness (M5.4 residual) ---------------------------------------


def fake_judge(faith=0.8):
    """A judge scorer returning a fixed faithfulness per record (no DeepEval / endpoint)."""

    def scorer(records, _model):
        return [{"faithfulness": faith, "answer_relevancy": 0.0} for _ in records]

    return scorer


def test_run_gated_judge_gating():
    from llb.bench.common import run_gated_judge

    recs = [{"question": "q", "answer": "a", "contexts": ["c"]}]
    assert run_gated_judge(recs, judge_model="j", judge_rho=0.7, scorer=fake_judge()).trusted
    assert not run_gated_judge(recs, judge_model="j", judge_rho=0.2, scorer=fake_judge()).trusted
    assert not run_gated_judge(recs, judge_model=None, judge_rho=0.9, scorer=fake_judge()).trusted


def test_summarization_gated_judge_trusted_records_faithfulness(tmp_path):
    cases = [
        bench_sm.SummarizationCase("a", "doc1", "ref1"),
        bench_sm.SummarizationCase("b", "doc2", "ref2"),
    ]
    run = bench_sm.run_summarization(
        cases,
        model="m",
        backend="ollama",
        complete=lambda _: "x",  # coverage stays low; faithfulness comes from the (fake) judge
        similarity=make_similarity({}),
        judge_model="judge",
        judge_rho=0.7,  # >= 0.6 -> trusted
        judge_scorer=fake_judge(0.8),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    assert run.judge_trusted is True
    assert run.faithfulness == 0.8 and run.faithfulness_ci is not None
    assert all(row["faithfulness"] == 0.8 for row in run.rows)
    # the headline stays OBJECTIVE coverage -- faithfulness is NOT folded in
    assert run.result.objective_score == 0.0


def test_summarization_gated_judge_below_threshold_is_demoted():
    cases = [bench_sm.SummarizationCase("a", "doc", "ref")]
    run = bench_sm.run_summarization(
        cases,
        model="m",
        backend="ollama",
        complete=lambda _: "ref",
        similarity=make_similarity({}),
        judge_model="judge",
        judge_rho=0.3,  # < 0.6 -> demoted
        judge_scorer=fake_judge(0.9),
        persist=False,
    )
    assert run.judge_trusted is False and run.faithfulness is None
    assert "faithfulness" not in run.rows[0]


def test_summarization_no_judge_is_objective_only():
    cases = [bench_sm.SummarizationCase("a", "doc", "ref")]
    run = bench_sm.run_summarization(
        cases,
        model="m",
        backend="ollama",
        complete=lambda _: "ref",
        similarity=make_similarity({}),
        persist=False,
    )
    assert run.judge_trusted is False and run.faithfulness is None
    assert run.judge_reason == "no judge configured"
