"""`llb check-run-bundle`: an operator's view of whether a published run can be handed on."""

from pathlib import Path

import pytest
import typer

from llb.cli.eval.run_bundles import check_run_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUNDLES = PROJECT_ROOT / "samples" / "artifact_contracts" / "run_bundles"


def test_a_current_bundle_reports_every_declared_member(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check_run_bundle(BUNDLES / "current")

    out = capsys.readouterr().out
    assert "5/5 member(s) readable" in out
    assert "2 record(s) of llb.case-score@1.0.0" in out
    assert "1 record(s) of llb.study-design@1.0.0" in out
    assert "opaque member, digest matches" in out


def test_a_pre_contract_bundle_reports_the_version_it_would_be_read_at(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check_run_bundle(BUNDLES / "legacy")

    out = capsys.readouterr().out
    assert "llb.run-manifest@1.0.0 -> 2.0.0" in out
    assert "3/3 member(s) readable" in out


def test_a_refused_member_exits_one_and_names_what_it_found(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as exit_info:
        check_run_bundle(BUNDLES / "mixed-version")

    assert exit_info.value.exit_code == 1
    out = capsys.readouterr().out
    assert "[refused] scores" in out
    assert "llb.agentic-case" in out


def test_an_unreadable_head_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit) as exit_info:
        check_run_bundle(BUNDLES / "unsupported-future")

    assert exit_info.value.exit_code == 2
    assert "version is not supported" in capsys.readouterr().err
