"""`llb check-bundle`: an operator's view of whether a bundle can be handed on."""

import json
import shutil
from pathlib import Path

import pytest
import typer

from llb.cli.prep.artifact_bundles import check_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "samples" / "artifact_contracts" / "data_prep"


def test_a_current_draft_bundle_reports_every_member(capsys: pytest.CaptureFixture[str]) -> None:
    check_bundle(FIXTURES / "draft-bundle", kind="draft", upgrade=False)

    out = capsys.readouterr().out
    assert "5/5 member(s) readable" in out
    assert "llb.gold-item@2.0.0" in out


def test_a_staged_corpus_reports_its_manifest_and_overlay(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check_bundle(FIXTURES / "corpus", kind="corpus", upgrade=False)

    assert "2/2 member(s) readable" in capsys.readouterr().out


def test_upgrade_rewrites_only_the_members_an_older_writer_produced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "draft"
    shutil.copytree(FIXTURES / "draft-bundle", bundle)
    shutil.copy(FIXTURES / "legacy" / "goldset.jsonl", bundle / "goldset.jsonl")
    ontology_before = (bundle / "ontology.json").read_bytes()

    check_bundle(bundle, kind="draft", upgrade=True)

    out = capsys.readouterr().out
    assert "upgraded 1 member(s): gold-items" in out
    assert (bundle / "ontology.json").read_bytes() == ontology_before


def test_a_member_this_build_cannot_read_exits_non_zero(tmp_path: Path) -> None:
    bundle = tmp_path / "draft"
    shutil.copytree(FIXTURES / "draft-bundle", bundle)
    record = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    record["schema_version"] = "9.0.0"
    (bundle / "provenance.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(typer.Exit) as refusal:
        check_bundle(bundle, kind="draft", upgrade=False)

    assert refusal.value.exit_code == 1


def test_an_unknown_kind_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as refusal:
        check_bundle(tmp_path, kind="stores", upgrade=False)

    assert refusal.value.exit_code == 2
