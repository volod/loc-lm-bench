"""The shared redistribution check (`llb.prep.corpus.redistribution`) and its CLI boundary."""

import json
from pathlib import Path

import pytest
import typer

from llb.core.runtime import REFUSAL_EXIT, run_typer
from llb.prep.corpus.redistribution import (
    LOCAL_ONLY,
    REDISTRIBUTABLE,
    RedistributionRefused,
    bundle_licences,
    chunk_licences,
    corpus_licences,
    merge_licences,
    permits_redistribution,
    refuse_non_redistributable,
)
from tests.llb.prep.redistribution_fixtures import (
    FIXTURE_ROOT,
    LOCAL_ONLY_DOC,
    staged_projection,
)


@pytest.mark.parametrize(
    ("licence", "permitted"),
    [
        (None, True),
        (REDISTRIBUTABLE, True),
        (LOCAL_ONLY, False),
        ("cc-by-4.0", False),  # a class this side cannot read is not a "yes"
        ("", False),
        ("Redistributable", False),
    ],
)
def test_only_absent_terms_and_the_redistributable_class_permit_egress(licence, permitted):
    assert permits_redistribution(licence) is permitted


def test_a_refusal_names_every_offending_document_in_order():
    licences = {"b.md": LOCAL_ONLY, "a.md": "unknown-class", "c.md": REDISTRIBUTABLE, "d.md": None}

    with pytest.raises(RedistributionRefused) as caught:
        refuse_non_redistributable(licences, egress="test export")

    assert caught.value.doc_ids == ("a.md", "b.md")
    assert caught.value.egress == "test export"
    message = str(caught.value)
    assert "a.md (licence=unknown-class)" in message
    assert "b.md (licence=local-only)" in message
    assert "refused rather than filtered" in message
    assert isinstance(caught.value, ValueError)


def test_the_check_is_scoped_to_what_the_path_carries():
    licences = {"local.md": LOCAL_ONLY, "open.md": REDISTRIBUTABLE}

    refuse_non_redistributable(licences, egress="scoped", doc_ids=["open.md", "unrecorded.md"])
    with pytest.raises(RedistributionRefused):
        refuse_non_redistributable(licences, egress="scoped", doc_ids=["local.md"])


def test_a_merge_keeps_the_strictest_terms_from_any_source():
    merged = merge_licences(
        {"a.md": REDISTRIBUTABLE, "b.md": None, "c.md": LOCAL_ONLY},
        {"a.md": LOCAL_ONLY, "b.md": REDISTRIBUTABLE, "c.md": REDISTRIBUTABLE},
    )
    assert merged == {"a.md": LOCAL_ONLY, "b.md": REDISTRIBUTABLE, "c.md": LOCAL_ONLY}


def test_chunk_licences_read_the_metadata_the_store_carries():
    chunks = [
        {"doc_id": "a.md", "metadata": {"licence": REDISTRIBUTABLE}},
        {"doc_id": "a.md", "metadata": {"licence": LOCAL_ONLY}},
        {"doc_id": "b.md"},
    ]
    assert chunk_licences(chunks) == {"a.md": LOCAL_ONLY, "b.md": None}


def test_the_staged_projection_records_exactly_one_refused_document(tmp_path: Path):
    licences = corpus_licences(staged_projection(tmp_path))

    assert len(licences) == len(list(FIXTURE_ROOT.glob("*.md")))
    with pytest.raises(RedistributionRefused) as caught:
        refuse_non_redistributable(licences, egress="whole corpus")
    assert caught.value.doc_ids == (LOCAL_ONLY_DOC,)


def test_a_corpus_without_a_manifest_or_licence_field_is_not_gated(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "doc.md").write_text("alpha\n", encoding="utf-8")

    assert corpus_licences(plain) == {}
    refuse_non_redistributable(corpus_licences(plain), egress="plain corpus")


def test_bundle_licences_read_the_provenance_document_rows(tmp_path: Path):
    (tmp_path / "provenance.json").write_text(
        json.dumps({"documents": [{"doc_id": "a.md", "licence": LOCAL_ONLY}, {"doc_id": "b.md"}]}),
        encoding="utf-8",
    )
    assert bundle_licences(tmp_path) == {"a.md": LOCAL_ONLY, "b.md": None}
    assert bundle_licences(tmp_path / "missing") == {}


def test_the_cli_boundary_reports_a_refusal_as_one_line_not_a_crash(monkeypatch, capsys):
    app = typer.Typer()

    @app.command()
    def export() -> None:
        refuse_non_redistributable({LOCAL_ONLY_DOC: LOCAL_ONLY}, egress="cli export")

    monkeypatch.setattr("sys.argv", ["llb"])
    with pytest.raises(SystemExit) as exited:
        run_typer(app)

    assert exited.value.code == REFUSAL_EXIT
    err = capsys.readouterr().err
    assert "[error] [redistribution] cli export refused: 1 document(s)" in err
    assert LOCAL_ONLY_DOC in err
    assert "Traceback" not in err and "FAILED" not in err
