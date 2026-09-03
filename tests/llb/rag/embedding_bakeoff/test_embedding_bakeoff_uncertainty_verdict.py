"""Paired sampling uncertainty of the embedder bake-off (`embedding_bakeoff_uncertainty`).

Pure: per-item metric vectors, the shared-index paired bootstrap, the adopt-or-retain verdict, and
the report columns all run over fake stores and plain vectors -- no FAISS, no GPU, no numpy.
"""

from llb.rag.embedding_bakeoff.models import BuiltStore


from llb.rag.embedding_bakeoff.report import format_report, render_markdown


from llb.rag.embedding_bakeoff.uncertainty import (
    METRIC_RECALL,
    paired_rows,
    recall_delta,
)


from llb.rag.embedding_bakeoff.verdict import (
    DECISION_ADOPT,
    DECISION_RETAIN,
    DECISION_UNDECIDED,
    decide_verdict,
)
from llb.rag.embedding_bakeoff.selection import adjust_bakeoff_selection
from llb.rag.comparison.sidecar import sidecar_report


from tests.llb.rag._embedding_bakeoff_uncertainty_helpers import (
    BASELINE,
    CLI_CANDIDATE,
    _HitSetStore,
    _questions,
    _vectors,
    _bakeoff,
)


def test_verdict_adopts_the_largest_separated_candidate():
    baseline = [1.0] * 4 + [0.0] * 16
    strong = [1.0] * 14 + [0.0] * 6  # +10 items
    weaker = [1.0] * 12 + [0.0] * 8  # +8 items, also separated
    paired = paired_rows(
        {
            BASELINE: _vectors(baseline),
            "strong": _vectors(strong),
            "weaker": _vectors(weaker),
        },
        BASELINE,
        resamples=500,
    )
    verdict = decide_verdict(paired, BASELINE)
    assert verdict["decision"] == DECISION_ADOPT
    assert verdict["model"] == "strong"
    assert verdict["separated"] == ["strong", "weaker"]
    assert "recall_at_k delta" in verdict["reason"]
    assert verdict["bars"] == ["recall_at_k"]
    assert verdict["cleared"] == {"strong": ["recall_at_k"], "weaker": ["recall_at_k"]}


def test_candidate_adoption_must_survive_the_selected_roster_family():
    baseline = [0.0] * 7
    vectors = {BASELINE: _vectors(baseline)}
    for loss in range(4):
        values = [-0.5 if index == loss else 1.0 for index in range(7)]
        vectors[f"candidate-{loss}"] = _vectors(values)
    paired = paired_rows(vectors, BASELINE, resamples=200)
    adjustment = adjust_bakeoff_selection(
        vectors,
        BASELINE,
        (METRIC_RECALL,),
        resamples=200,
        seed=13,
    )
    verdict = decide_verdict(paired, BASELINE, adjustment=adjustment)

    assert verdict["per_row_cleared"] == {
        f"candidate-{index}": [METRIC_RECALL] for index in range(4)
    }
    assert verdict["decision"] == DECISION_RETAIN
    assert verdict["selection_adjustment"]["family_size"] == 4


def test_verdict_retains_the_incumbent_when_nothing_separates():
    paired = paired_rows(
        {
            BASELINE: _vectors([1.0, 0.0, 1.0, 0.0]),
            "cand": _vectors([1.0, 1.0, 0.0, 0.0]),  # one win, one loss
        },
        BASELINE,
        resamples=200,
    )
    verdict = decide_verdict(paired, BASELINE)
    assert verdict["decision"] == DECISION_RETAIN
    assert verdict["model"] == BASELINE and verdict["separated"] == []


def test_verdict_is_undecided_without_a_baseline():
    assert decide_verdict({}, None)["decision"] == DECISION_UNDECIDED
    assert decide_verdict({}, BASELINE)["decision"] == DECISION_UNDECIDED


def test_run_bakeoff_carries_a_paired_interval_on_every_candidate_row():
    report = _bakeoff()
    assert report["uncertainty"]["baseline"] == BASELINE
    assert report["uncertainty"]["resamples"] == 500
    by_model = {row["model"]: row for row in report["candidates"]}
    assert set(by_model) == {BASELINE, "cand"}
    for row in by_model.values():
        assert row["paired_vs_baseline"]["baseline"] == BASELINE
    assert recall_delta(by_model["cand"]["paired_vs_baseline"])["wins"] == 10
    assert report["verdict"]["decision"] == DECISION_ADOPT
    assert report["verdict"]["model"] == "cand"


def test_run_bakeoff_without_a_baseline_leaves_the_rows_bare():
    report = _bakeoff(baseline=None)
    assert all("paired_vs_baseline" not in row for row in report["candidates"])
    assert report["verdict"]["decision"] == DECISION_UNDECIDED
    md = render_markdown(report)
    assert "UNDECIDED" in md and md.isascii()


def test_report_renders_the_delta_column_the_ledger_and_the_verdict():
    report = _bakeoff()
    md = render_markdown(report)
    assert f"| recall delta vs {BASELINE} | w/l/t | sign p |" in md
    assert "10/0/10" in md  # the item-level ledger behind the interval
    assert "Verdict: ADOPT `cand`" in md
    assert "paired uncertainty: baseline" in md
    assert md.isascii()  # AGENTS.md: ASCII-only output
    text = format_report(report)
    assert "d_recall vs baseline" in text and "Verdict: ADOPT" in text
    assert text.isascii()


def test_cli_writes_the_paired_ledger_machine_readable(tmp_path, monkeypatch):
    """`compare-embeddings` persists report.json beside report.md, intervals included.

    The recorded recommendation could not be re-read because only prose survived; the JSON is what
    a later re-read recomputes from.
    """

    from typer.testing import CliRunner

    from llb.cli.app import app
    from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset(
        [
            GoldItem(
                id=f"item-{i}",
                question=question,
                reference_answer="x",
                source_doc_id="d1",
                source_spans=[
                    SourceSpan(doc_id="d1", char_start=0, char_end=10, text="0123456789")
                ],
                provenance="ontology-drafted",
                split="final",
            )
            for i, question in enumerate(_questions(20))
        ],
        goldset,
    )
    stores = {
        BASELINE: _HitSetStore(set(_questions(20)[:4])),
        CLI_CANDIDATE: _HitSetStore(set(_questions(20)[:14])),
    }
    monkeypatch.setattr(
        "llb.cli.rag.compare_embeddings.local_store_builder",
        lambda cfg, stores_dir, **_kwargs: (
            lambda model: BuiltStore(store=stores[model], embed_seconds=1.0, index_bytes=100)
        ),
    )
    out = tmp_path / "report.md"
    result = CliRunner().invoke(
        app,
        [
            "compare-embeddings",
            "--goldset",
            str(goldset),
            "--corpus-root",
            str(corpus),
            "--models",
            f"{BASELINE},{CLI_CANDIDATE}",
            "--k",
            "1",
            "--baseline",
            BASELINE,
            "--resamples",
            "200",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    report = sidecar_report(out.with_suffix(".json"))
    assert report["uncertainty"]["baseline"] == BASELINE
    assert (
        report["verdict"]["decision"] == DECISION_ADOPT
        and report["verdict"]["model"] == CLI_CANDIDATE
    )
    row = next(r for r in report["candidates"] if r["model"] == CLI_CANDIDATE)
    assert row["paired_vs_baseline"]["metrics"][METRIC_RECALL]["wins"] == 10
