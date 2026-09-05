"""`llb check-generation`: an operator's view of whether a built generation can be queried."""

import shutil
from pathlib import Path

import pytest
import typer

from llb.cli.rag.artifact_generations import check_generation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "samples" / "artifact_contracts" / "retrieval_graph"


def test_a_current_store_reports_every_member_including_the_opaque_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check_generation(FIXTURES / "store" / "current", kind="store")

    out = capsys.readouterr().out
    assert "5/5 member(s) readable" in out
    assert "2 record(s) of llb.rag-chunk@1.0.0" in out
    assert "opaque member, digest matches" in out


def test_a_pre_contract_store_reports_the_version_it_would_be_read_at(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check_generation(FIXTURES / "store" / "legacy", kind="store")

    out = capsys.readouterr().out
    assert "3/3 member(s) readable" in out
    assert "llb.rag-store-meta@1.0.0 -> 2.0.0" in out


def test_a_graph_and_a_prompt_system_package_report_their_own_members(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check_generation(FIXTURES / "graph" / "current", kind="graph")
    assert "4/4 member(s) readable" in capsys.readouterr().out

    check_generation(FIXTURES / "prompt_system" / "legacy", kind="prompt-system")
    assert "5/5 member(s) readable" in capsys.readouterr().out


def test_a_member_this_build_cannot_read_exits_one_and_names_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "store"
    shutil.copytree(FIXTURES / "store" / "current", store)
    (store / "index.faiss").write_bytes(b"a different faiss index\n")

    with pytest.raises(typer.Exit) as excinfo:
        check_generation(store, kind="store")

    assert excinfo.value.exit_code == 1
    assert "[refused] vector-index" in capsys.readouterr().out


def test_an_unknown_kind_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit) as excinfo:
        check_generation(FIXTURES / "store" / "current", kind="bundle")

    assert excinfo.value.exit_code == 2
    assert "unknown --kind" in capsys.readouterr().err
