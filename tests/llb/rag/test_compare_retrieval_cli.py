"""GraphRAG backend residual 3 -- graph-vs-FAISS retrieval comparison core (`llb.rag.compare`).

Pure: driven by fake stores exposing the `.retrieve` seam, so it runs in the lightweight CI install
(no FAISS, no DuckDB, no GPU). The CLI wiring (`compare-retrieval`) layers real stores on top.
"""

from llb.cli.rag.compare_stores import _compare_vector_corpus_root


from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset


from llb.rag.question_types import (
    aligned_question_types,
    load_question_types,
    load_question_types_by_question,
)


from _compare_retrieval_helpers import (
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


def test_question_type_labels_find_parent_sidecar_for_accepted_ledger(tmp_path):
    import json

    accepted = tmp_path / "accepted"
    accepted.mkdir()
    goldset = accepted / "goldset.jsonl"
    goldset.write_text("", encoding="utf-8")
    rows = [
        {"id": "a", "question_type": "comparative"},
        {"id": "b", "question_type": "multi-hop"},
    ]
    (tmp_path / "needle_items.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    assert aligned_question_types(goldset, ["b", "missing", "a"]) == [
        "multi-hop",
        None,
        "comparative",
    ]
    assert load_question_types(goldset) == {"a": "comparative", "b": "multi-hop"}


def test_question_type_labels_are_absent_without_a_needle_sidecar(tmp_path):
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text("", encoding="utf-8")
    assert aligned_question_types(goldset, ["a"]) is None
    assert load_question_types(goldset) == {}


def test_question_type_map_omits_duplicate_question_text_with_conflicting_labels(tmp_path):
    goldset = tmp_path / "goldset.jsonl"
    items = [
        GoldItem(
            id=item_id,
            question=question,
            reference_answer="x",
            source_doc_id="d",
            source_spans=[SourceSpan(doc_id="d", char_start=0, char_end=1, text="x")],
            provenance="ontology-drafted",
            split="final",
        )
        for item_id, question in (("a", "same"), ("b", "same"), ("c", "unique"))
    ]
    dump_goldset(items, goldset)
    (tmp_path / "needle_items.jsonl").write_text(
        "\n".join(
            [
                '{"id":"a","question_type":"factoid"}',
                '{"id":"b","question_type":"multi-hop"}',
                '{"id":"c","question_type":"comparative"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_question_types_by_question(goldset) == {"unique": "comparative"}


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
        "llb.rag.comparison_builders.build_vector_store_comparison",
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


def test_compare_retrieval_cli_persists_paired_rows_and_mode_baseline(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from llb.main import app

    bundle = tmp_path / "bundle"
    (bundle / "corpus").mkdir(parents=True)
    goldset = bundle / "goldset.jsonl"
    dump_goldset(
        [
            GoldItem(
                id="paired-a",
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
    monkeypatch.setattr(
        "llb.cli.rag.compare_retrieval._build_compare_stores",
        lambda cfg, strategies, hybrid, compare_items: {
            "sentence": _FakeStore([_chunk("d1", 0, 10)]),
            "recursive": _FakeStore([_chunk("d1", 0, 10)]),
        },
    )
    out = tmp_path / "nested" / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "compare-retrieval",
            "--goldset",
            str(goldset),
            "--strategies",
            "sentence,recursive",
            "--resamples",
            "50",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Verdict: RETAIN `recursive`" in result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["uncertainty"]["baseline"] == "recursive"
    assert report["paired_items"][0]["item_id"] == "paired-a"
    assert "paired_vs_baseline" in report["backends"]["sentence"]
