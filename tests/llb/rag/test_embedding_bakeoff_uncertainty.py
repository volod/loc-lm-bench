"""Paired sampling uncertainty of the embedder bake-off (`embedding_bakeoff_uncertainty`).

Pure: per-item metric vectors, the shared-index paired bootstrap, the adopt-or-retain verdict, and
the report columns all run over fake stores and plain vectors -- no FAISS, no GPU, no numpy.
"""

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.embedding_bakeoff import run_bakeoff, score_candidate
from llb.rag.embedding_bakeoff_models import BuiltStore
from llb.rag.embedding_bakeoff_report import format_report, render_markdown
from llb.rag.embedding_bakeoff_uncertainty import (
    DECISION_ADOPT,
    DECISION_RETAIN,
    DECISION_UNDECIDED,
    METRIC_MRR,
    METRIC_RECALL,
    decide_verdict,
    item_vectors,
    paired_rows,
    recall_delta,
    separates_from_baseline,
)

BASELINE = "baseline-model"


def _chunk(doc: str, start: int) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": start + 10, "text": "x"}


def _span(doc: str) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": 0, "char_end": 10, "text": "g"}


class _HitSetStore:
    """Retrieves the gold chunk for the questions in `hits`, and a miss for every other."""

    def __init__(self, hits: set[str]):
        self._hits = hits
        self.meta = {"dim": 8, "n_indexed": 3, "embedding_model": "m"}

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        gold = _chunk("d1", 0)
        miss = _chunk("d9", 500)
        return ([gold, miss] if question in self._hits else [miss, gold])[:k]


def _questions(n: int) -> list[str]:
    return [f"питання-{i:02d}" for i in range(n)]


def _items(n: int) -> list[tuple[str, list[SourceSpanRecord]]]:
    return [(question, [_span("d1")]) for question in _questions(n)]


def _vectors(recall: list[float], mrr: list[float] | None = None) -> dict[str, list[float]]:
    return {METRIC_RECALL: recall, METRIC_MRR: mrr if mrr is not None else list(recall)}


def test_item_vectors_mean_matches_the_published_row():
    items = _items(4)
    store = _HitSetStore({items[0][0], items[2][0]})
    built = BuiltStore(store=store, embed_seconds=1.0, index_bytes=10)
    row = score_candidate("m", built, items, k=2)
    vectors = item_vectors([(store.retrieve(q, 2), spans) for q, spans in items], k=2)
    assert vectors[METRIC_RECALL] == [1.0, 1.0, 1.0, 1.0]  # k=2 retrieves both chunks
    assert vectors[METRIC_MRR] == [1.0, 0.5, 1.0, 0.5]  # gold first only where the store hits
    assert row["recall_at_k"] == sum(vectors[METRIC_RECALL]) / 4
    assert row["mrr"] == sum(vectors[METRIC_MRR]) / 4


def test_paired_delta_keeps_the_item_pairing():
    # Candidate wins 8 items outright and never loses one: a consistent, separated lead.
    baseline = [1.0] * 6 + [0.0] * 14
    candidate = [1.0] * 6 + [1.0] * 8 + [0.0] * 6
    paired = paired_rows(
        {BASELINE: _vectors(baseline), "cand": _vectors(candidate)}, BASELINE, resamples=500
    )
    delta = recall_delta(paired["cand"])
    assert (delta["wins"], delta["losses"], delta["ties"]) == (8, 0, 12)
    assert delta["delta"]["mean"] == 0.4
    assert delta["delta"]["lo"] > 0.0  # the interval clears zero -> a separated candidate
    assert separates_from_baseline(paired["cand"]) is True
    assert paired["cand"]["baseline"] == BASELINE


def test_baseline_row_is_paired_against_itself_at_exactly_zero():
    paired = paired_rows({BASELINE: _vectors([1.0, 0.0, 1.0])}, BASELINE, resamples=64)
    delta = recall_delta(paired[BASELINE])["delta"]
    assert (delta["mean"], delta["lo"], delta["hi"]) == (0.0, 0.0, 0.0)
    assert separates_from_baseline(paired[BASELINE]) is False


def test_a_one_item_lead_does_not_separate():
    # The exact shape the bake-off re-read produced: a two-question lead on a 40-item set.
    baseline = [1.0] * 37 + [0.0] * 3
    candidate = [1.0] * 39 + [0.0]
    paired = paired_rows(
        {BASELINE: _vectors(baseline), "cand": _vectors(candidate)}, BASELINE, resamples=500
    )
    assert recall_delta(paired["cand"])["delta"]["lo"] == 0.0  # touches zero -> not separated
    assert separates_from_baseline(paired["cand"]) is False


def test_paired_rows_share_one_index_set_and_are_seed_deterministic():
    vectors = {
        BASELINE: _vectors([1.0, 0.0] * 10),
        "cand": _vectors([1.0] * 14 + [0.0] * 6),
    }
    first = paired_rows(vectors, BASELINE, resamples=200, seed=7)
    again = paired_rows(vectors, BASELINE, resamples=200, seed=7)
    other_seed = paired_rows(vectors, BASELINE, resamples=200, seed=8)
    assert first == again  # same seed -> byte-identical intervals
    assert recall_delta(first["cand"])["delta"] != recall_delta(other_seed["cand"])["delta"]


def test_paired_rows_are_empty_when_the_baseline_was_not_scored():
    assert paired_rows({"cand": _vectors([1.0])}, BASELINE) == {}


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
    assert "paired recall@k delta" in verdict["reason"]


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


def _bakeoff(baseline: str | None = BASELINE, n: int = 20):
    items = _items(n)
    questions = _questions(n)
    stores = {
        BASELINE: _HitSetStore(set(questions[:4])),
        "cand": _HitSetStore(set(questions[:14])),
    }
    return run_bakeoff(
        items,
        k=1,  # k=1 so a miss is a miss: the store puts the gold chunk second
        corpus_root="corpus",
        local_models=[BASELINE, "cand"],
        build_local=lambda model: BuiltStore(
            store=stores[model], embed_seconds=1.0, index_bytes=100
        ),
        baseline=baseline,
        resamples=500,
    )


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
    import json

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
        "cand": _HitSetStore(set(_questions(20)[:14])),
    }
    monkeypatch.setattr(
        "llb.cli.rag.compare_embeddings._local_store_builder",
        lambda cfg, stores_dir: (
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
            f"{BASELINE},cand",
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
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["uncertainty"]["baseline"] == BASELINE
    assert report["verdict"]["decision"] == DECISION_ADOPT and report["verdict"]["model"] == "cand"
    row = next(r for r in report["candidates"] if r["model"] == "cand")
    assert row["paired_vs_baseline"]["metrics"][METRIC_RECALL]["wins"] == 10
