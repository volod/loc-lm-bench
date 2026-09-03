"""`llb check-store`: an operator's view of a store, a graph, or a prompt-system package."""

import json
import shutil
import sys
from pathlib import Path

import pytest
import typer

from llb.cli.rag.check_store import check_store

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "samples" / "artifact_contracts" / "retrieval_graph"


def test_a_current_store_reports_its_rows_and_names_its_opaque_members(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check_store(FIXTURES / "store", kind="store", upgrade=False)

    out = capsys.readouterr().out
    assert "4/4 member(s) readable" in out
    assert "2 record(s) of llb.rag-chunk@1.0.0" in out
    assert "opaque, owned by faiss@IndexFlatIP/1" in out
    assert "opaque, owned by llb.rag.vector_store.lexical_index@bm25-uk-v2" in out


def test_a_graph_and_a_package_report_every_member(capsys: pytest.CaptureFixture[str]) -> None:
    check_store(FIXTURES / "graph", kind="graph", upgrade=False)
    assert "4/4 member(s) readable" in capsys.readouterr().out

    check_store(FIXTURES / "prompt-system", kind="prompt-system", upgrade=False)
    assert "5/5 member(s) readable" in capsys.readouterr().out


def test_a_pre_contract_store_reads_at_the_only_version_and_upgrades_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every retrieval family is at its initial version, so there is nothing to carry forward."""
    store = tmp_path / "store"
    shutil.copytree(FIXTURES / "legacy" / "store", store)
    before = {path.name: path.read_bytes() for path in sorted(store.iterdir())}

    check_store(store, kind="store", upgrade=True)

    out = capsys.readouterr().out
    assert "upgraded 0 member(s):" in out
    assert "2/2 member(s) readable" in out
    assert {path.name: path.read_bytes() for path in sorted(store.iterdir())} == before


def test_a_member_this_build_cannot_read_exits_non_zero(tmp_path: Path) -> None:
    store = tmp_path / "store"
    shutil.copytree(FIXTURES / "store", store)
    record = json.loads((store / "store_meta.json").read_text(encoding="utf-8"))
    (store / "store_meta.json").write_text(
        json.dumps({**record, "schema_version": "9.0.0"}), encoding="utf-8"
    )

    with pytest.raises(typer.Exit) as refusal:
        check_store(store, kind="store", upgrade=False)

    assert refusal.value.exit_code == 1


def test_an_unknown_kind_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as refusal:
        check_store(tmp_path, kind="bundle", upgrade=False)

    assert refusal.value.exit_code == 2


def test_a_directory_with_no_registered_member_exits_two(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as refusal:
        check_store(tmp_path, kind="store", upgrade=False)

    assert refusal.value.exit_code == 2


def test_inspection_imports_neither_faiss_nor_duckdb_nor_an_encoder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The point of the command: a store can be read on a host that could not query it."""
    check_store(FIXTURES / "store", kind="store", upgrade=False)
    check_store(FIXTURES / "graph", kind="graph", upgrade=False)
    capsys.readouterr()

    assert not {"faiss", "duckdb", "sentence_transformers"} & set(sys.modules)
