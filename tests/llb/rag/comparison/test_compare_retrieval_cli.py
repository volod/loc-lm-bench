"""`compare-retrieval` CLI wiring: the question-type sidecar join and the persisted paired report.

Pure: driven by fake stores exposing the `.retrieve` seam, so it runs in the lightweight CI install
(no FAISS, no DuckDB, no GPU). The scoring core is covered in `test_compare_retrieval_core.py` and
the backend command in `test_compare_vector_stores_cli.py`.
"""

from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset


from llb.rag.comparison.sidecar import sidecar_report
from llb.rag.question_types import (
    aligned_question_types,
    load_question_types,
    load_question_types_by_question,
)


from tests.llb.rag._compare_retrieval_helpers import (
    _FakeStore,
    _chunk,
    _exact_chunk,
)


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


def test_question_type_labels_come_from_the_item_provenance_sidecar(tmp_path):
    # The external-draft import lane writes `item_provenance.jsonl` instead of a needle sidecar.
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text("", encoding="utf-8")
    (tmp_path / "item_provenance.jsonl").write_text(
        '{"id":"a","question_type":"numeric"}\n', encoding="utf-8"
    )
    assert load_question_types(goldset) == {"a": "numeric"}
    assert aligned_question_types(goldset, ["a", "b"]) == ["numeric", None]


def test_question_type_sidecars_join_with_the_needle_sidecar_winning(tmp_path):
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text("", encoding="utf-8")
    (tmp_path / "needle_items.jsonl").write_text(
        '{"id":"a","question_type":"numeric"}\n', encoding="utf-8"
    )
    (tmp_path / "item_provenance.jsonl").write_text(
        '{"id":"a","question_type":"factoid"}\n{"id":"b","question_type":"comparative"}\n',
        encoding="utf-8",
    )
    assert load_question_types(goldset) == {"a": "numeric", "b": "comparative"}


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


def test_compare_retrieval_cli_persists_paired_rows_and_mode_baseline(tmp_path, monkeypatch):

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
        "llb.cli.rag.compare_retrieval_lanes.build_compare_stores",
        lambda cfg, strategies, sizes, hybrid, compare_items: {
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
    report = sidecar_report(out)
    assert report["uncertainty"]["baseline"] == "recursive"
    assert report["paired_items"][0]["item_id"] == "paired-a"
    assert "paired_vs_baseline" in report["backends"]["sentence"]


def _paired_goldset(tmp_path):
    """One bundle whose single gold span is CUT across the two chunks the fake store returns."""
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
                    SourceSpan(doc_id="d1", char_start=40, char_end=60, text="01234567890123456789")
                ],
                provenance="ontology-drafted",
                split="final",
            )
        ],
        goldset,
    )
    return goldset


def test_compare_retrieval_cli_stitch_twin_is_reported_but_never_adopted(tmp_path, monkeypatch):

    from typer.testing import CliRunner

    from llb.main import app

    goldset = _paired_goldset(tmp_path)
    monkeypatch.setattr(
        "llb.cli.rag.compare_retrieval_lanes.build_compare_stores",
        lambda cfg, strategies, sizes, hybrid, compare_items: {
            "recursive": _FakeStore([_exact_chunk("d1", 0, 50), _exact_chunk("d1", 50, 100)])
        },
    )
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "compare-retrieval",
            "--goldset",
            str(goldset),
            "--strategies",
            "recursive",
            "--stitch",
            "--resamples",
            "50",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    report = sidecar_report(out)
    base = report["backends"]["recursive"]
    stitched = report["backends"]["recursive+stitch"]
    assert base["span_intact_at_k"] == 0.0 and stitched["span_intact_at_k"] == 1.0
    assert stitched["recall_at_k"] == base["recall_at_k"]
    assert report["stitching"]["recursive+stitch"]["recall_invariant"] is True
    # the twin is a reported lever, so no verdict can name it
    assert "recursive+stitch" not in report["uncertainty"]["eligible_lanes"]
    assert report["verdict"]["lane"] != "recursive+stitch"


def test_compare_retrieval_cli_size_lanes_are_paired_against_the_configs_own_size(
    tmp_path, monkeypatch
):

    from typer.testing import CliRunner

    from llb.main import app

    goldset = _paired_goldset(tmp_path)
    monkeypatch.setattr(
        "llb.cli.rag.compare_retrieval_lanes.build_compare_stores",
        lambda cfg, strategies, sizes, hybrid, compare_items: {
            "recursive#size1600": _FakeStore([_exact_chunk("d1", 0, 100)]),
            "recursive#size800": _FakeStore(
                [_exact_chunk("d1", 0, 50), _exact_chunk("d1", 50, 100)]
            ),
        },
    )
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "compare-retrieval",
            "--goldset",
            str(goldset),
            "--sizes",
            "1600,800",
            "--resamples",
            "50",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    report = sidecar_report(out)
    # the config's shipped size (800) is the incumbent, not the first lane the operator listed
    assert report["uncertainty"]["baseline"] == "recursive#size800"
    assert report["backends"]["recursive#size1600"]["span_intact_at_k"] == 1.0
    assert report["backends"]["recursive#size800"]["span_intact_at_k"] == 0.0
    # and the served-context column prices each cap beside the gain it bought
    assert report["backends"]["recursive#size1600"]["served_chars_at_k"] == 100.0


def test_compare_retrieval_cli_refuses_two_comparison_modes_at_once(tmp_path):
    from typer.testing import CliRunner

    from llb.main import app

    goldset = _paired_goldset(tmp_path)
    result = CliRunner().invoke(
        app,
        ["compare-retrieval", "--goldset", str(goldset), "--sizes", "400", "--hybrid"],
    )
    assert result.exit_code == 2
    assert "--sizes, --hybrid are mutually exclusive" in result.output


def test_compare_retrieval_cli_refuses_a_non_integer_size(tmp_path):
    from typer.testing import CliRunner

    from llb.main import app

    goldset = _paired_goldset(tmp_path)
    result = CliRunner().invoke(
        app, ["compare-retrieval", "--goldset", str(goldset), "--sizes", "400,big"]
    )
    assert result.exit_code == 2
    assert "--sizes takes integers" in result.output
