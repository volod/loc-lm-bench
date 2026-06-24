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
