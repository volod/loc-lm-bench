"""`compare-vector-stores` CLI wiring: corpus inference, the floor, and the paired backend lane.

Pure: every backend is a fake store behind the `.retrieve` seam, so the whole command runs in the
lightweight CI install (no FAISS, no Chroma/Qdrant client, no GPU). The scoring core it drives is
covered in `test_compare_retrieval_core.py`.
"""

from llb.cli.rag.compare_stores import _compare_vector_corpus_root

from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset


from tests.llb.rag._compare_retrieval_helpers import (
    _FakeStore,
    _chunk,
)


def test_compare_vector_stores_infers_sibling_corpus(tmp_path):
    root = tmp_path / "bundle"
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    goldset = root / "goldset.jsonl"
    goldset.write_text("", encoding="utf-8")

    assert _compare_vector_corpus_root(goldset, None) == corpus
    explicit = tmp_path / "other-corpus"
    assert _compare_vector_corpus_root(goldset, explicit) == explicit
    assert _compare_vector_corpus_root(tmp_path / "missing" / "goldset.jsonl", None) is None


def test_compare_vector_stores_publishes_the_floor_when_asked(tmp_path, monkeypatch):
    """The backend lane reads the same floor as `compare-retrieval` (`--noise-floor`)."""
    import json

    from typer.testing import CliRunner

    from llb.main import app

    goldset = tmp_path / "goldset.jsonl"
    dump_goldset(
        [
            GoldItem(
                id="a",
                question="питання",
                reference_answer="x",
                source_doc_id="d1",
                source_spans=[
                    SourceSpan(doc_id="d1", char_start=0, char_end=10, text="0123456789")
                ],
                provenance="ontology-drafted",
                split="final",
            )
        ],
        goldset,
    )
    tied = [
        {**_chunk("d2", 20, 30), "retrieval_score": 0.5},
        {**_chunk("d1", 0, 10), "retrieval_score": 0.5},
    ]
    monkeypatch.setattr(
        "llb.rag.comparison.builders.build_vector_store_comparison",
        lambda cfg, backends: {name: _FakeStore(tied) for name in backends},
    )
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "compare-vector-stores",
            "--goldset",
            str(goldset),
            "--backends",
            "faiss,chroma",
            "--k",
            "1",
            "--noise-floor",
            "--noise-floor-replicates",
            "16",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "noise floor" in result.output
    floor = json.loads(out.read_text(encoding="utf-8"))["noise_floor"]
    assert set(floor["lanes"]) == {"faiss", "chroma"}
    # Both backends rank the same tie, so neither is distinguished from the other.
    assert floor["floor_recall_at_k"] > 0.0 and floor["margin"]["clears_floor"] is False


def _one_item_goldset(path, item_id="paired-a"):
    dump_goldset(
        [
            GoldItem(
                id=item_id,
                question="питання",
                reference_answer="x",
                source_doc_id="d1",
                source_spans=[
                    SourceSpan(doc_id="d1", char_start=0, char_end=10, text="0123456789")
                ],
                provenance="ontology-drafted",
                split="final",
            )
        ],
        path,
    )
    return path


def test_compare_vector_stores_pairs_every_backend_against_faiss(tmp_path, monkeypatch):
    """A backend swap is decided the way an embedder swap is: paired delta + adopt-or-retain."""
    import json

    from typer.testing import CliRunner

    from llb.main import app

    goldset = _one_item_goldset(tmp_path / "goldset.jsonl", "store-a")
    # chroma finds the span, faiss does not -- so the point-estimate leader is NOT the baseline
    # and the verdict has to say whether one scored item can resolve that gap.
    monkeypatch.setattr(
        "llb.rag.comparison.builders.build_vector_store_comparison",
        lambda cfg, backends: {
            "chroma": _FakeStore([_chunk("d1", 0, 10)]),
            "faiss": _FakeStore([_chunk("d2", 20, 30)]),
        },
    )
    out = tmp_path / "nested" / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "compare-vector-stores",
            "--goldset",
            str(goldset),
            "--backends",
            "chroma,faiss",
            "--k",
            "1",
            "--resamples",
            "50",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "paired vs faiss: 50 resamples" in result.output
    assert "Verdict: RETAIN `faiss`" in result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    # The baseline is the incumbent backend, not whichever label the build happened to emit first.
    assert report["uncertainty"]["baseline"] == "faiss"
    assert report["paired_items"][0]["item_id"] == "store-a"
    for backend in ("faiss", "chroma"):
        paired = report["backends"][backend]["paired_vs_baseline"]
        assert paired["baseline"] == "faiss"
        delta = paired["metrics"]["recall_at_k"]["delta"]
        assert delta["lo"] <= delta["mean"] <= delta["hi"]
        assert set(paired["metrics"]) == {
            "recall_at_k",
            "mrr",
            "span_char_coverage_at_k",
            "span_intact_at_k",
        }
    assert report["verdict"]["decision"] == "retain"


def test_compare_vector_stores_refuses_a_baseline_backend_it_did_not_score(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from llb.main import app

    goldset = _one_item_goldset(tmp_path / "goldset.jsonl", "store-b")
    monkeypatch.setattr(
        "llb.rag.comparison.builders.build_vector_store_comparison",
        lambda cfg, backends: {name: _FakeStore([_chunk("d1", 0, 10)]) for name in backends},
    )
    result = CliRunner().invoke(
        app,
        [
            "compare-vector-stores",
            "--goldset",
            str(goldset),
            "--backends",
            "faiss,chroma",
            "--baseline",
            "qdrant",
        ],
    )

    assert result.exit_code == 2
    assert "paired baseline lane `qdrant` was not scored" in result.output


def test_compare_vector_stores_falls_back_to_the_first_backend_without_faiss(tmp_path, monkeypatch):
    """Without the incumbent in the row set the paired lane still names one stable baseline."""
    import json

    from typer.testing import CliRunner

    from llb.main import app

    goldset = _one_item_goldset(tmp_path / "goldset.jsonl", "store-c")
    monkeypatch.setattr(
        "llb.rag.comparison.builders.build_vector_store_comparison",
        lambda cfg, backends: {name: _FakeStore([_chunk("d1", 0, 10)]) for name in backends},
    )
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "compare-vector-stores",
            "--goldset",
            str(goldset),
            "--backends",
            "chroma,qdrant",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8"))["uncertainty"]["baseline"] == "chroma"
