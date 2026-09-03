"""`llb check-run`: what an operator is told about a bundle before anything reads it."""

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from llb.artifacts.runs.datasets import KIND_BENCHMARK
from llb.core.contracts.run_bundle import CASE_SCORE_SCHEMA_ID
from llb.main import app

FIXTURES = Path(__file__).resolve().parents[3] / "samples" / "artifact_contracts" / "run_bundles"
CURRENT_RUN = FIXTURES / "run"
CURRENT_BENCHMARK = FIXTURES / "benchmark"
LEGACY_RUN = FIXTURES / "legacy" / "run"
UNSUPPORTED = FIXTURES / "unsupported-future"

runner = CliRunner()


def test_a_published_bundle_reports_every_member_and_its_contract():
    result = runner.invoke(app, ["check-run", str(CURRENT_RUN)])
    assert result.exit_code == 0, result.output
    assert "7/7 member(s) readable in llb-run-bundle" in result.output
    assert f"scores.jsonl): 2 record(s) of {CASE_SCORE_SCHEMA_ID}@1.0.0" in result.output
    assert "second-fold-analysis (second-fold-analysis.json): 1 record(s)" in result.output


def test_a_benchmark_bundle_reports_its_cells_under_the_cell_contract():
    result = runner.invoke(app, ["check-run", str(CURRENT_BENCHMARK), "--kind", KIND_BENCHMARK])
    assert result.exit_code == 0, result.output
    assert "llb.benchmark-cell@1.0.0" in result.output


def test_a_pre_contract_benchmark_bundle_needs_no_kind_flag(tmp_path):
    """It records the category it is a cell of, so the operator does not have to."""
    bundle = tmp_path / "legacy-benchmark"
    shutil.copytree(FIXTURES / "legacy" / "benchmark", bundle)
    result = runner.invoke(app, ["check-run", str(bundle)])
    assert result.exit_code == 0, result.output
    assert "llb.benchmark-cell@1.0.0" in result.output


def test_a_pre_contract_bundle_reads_and_needs_no_upgrade(tmp_path):
    bundle = tmp_path / "legacy-run"
    shutil.copytree(LEGACY_RUN, bundle)
    before = {path: path.read_bytes() for path in sorted(bundle.rglob("*")) if path.is_file()}

    result = runner.invoke(app, ["check-run", str(bundle), "--upgrade"])
    assert result.exit_code == 0, result.output
    assert "upgraded 0 member(s)" in result.output
    # Every family here is at its initial version, so `--upgrade` has nothing to rewrite: the
    # flag is the same one `check-bundle` and `check-store` carry, not a second implementation.
    assert {path: path.read_bytes() for path in before} == before


def test_a_future_major_is_named_rather_than_read():
    result = runner.invoke(app, ["check-run", str(UNSUPPORTED)])
    assert result.exit_code == 1
    assert "[refused] run-manifest" in result.output


def test_a_tampered_member_refuses_on_its_digest(tmp_path):
    bundle = tmp_path / "run"
    shutil.copytree(CURRENT_RUN, bundle)
    rows = (bundle / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["objective_score"] = 0.0
    (bundle / "scores.jsonl").write_text(
        "\n".join([json.dumps(first), *rows[1:]]) + "\n", encoding="utf-8"
    )
    result = runner.invoke(app, ["check-run", str(bundle)])
    assert result.exit_code == 1
    assert "digest mismatch" in result.output


def test_an_unknown_kind_is_refused_with_the_choices():
    result = runner.invoke(app, ["check-run", str(CURRENT_RUN), "--kind", "nonsense"])
    assert result.exit_code == 2
    assert "unknown --kind 'nonsense'" in result.output
