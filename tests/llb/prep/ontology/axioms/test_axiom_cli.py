"""The command publishes a readable run bundle, and the build refusal honours the sign-off boundary."""

import json

from typer.testing import CliRunner

from llb.cli import app
from llb.prep.ontology.axioms.constants import (
    AXIOM_EVIDENCE_FILENAME,
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    VIOLATIONS_FILENAME,
)
from llb.prep.ontology.axioms.loader import save_axioms
from llb.prep.ontology.axioms.models import AxiomSet
from llb.prep.ontology.axioms.run import check_build_inputs
from llb.prep.ontology.models import DocExtraction

from tests.llb.prep.ontology.axioms.conftest import CANDIDATE_TURTLE, FIXTURE_LEDGER

RUN = "test-run"


def _invoke(tmp_path, *extra: str):
    return CliRunner().invoke(
        app,
        [
            "validate-ontology-axioms",
            "--extraction",
            str(FIXTURE_LEDGER),
            "--axioms",
            str(CANDIDATE_TURTLE),
            "--data-dir",
            str(tmp_path),
            "--run",
            RUN,
            *extra,
        ],
    )


def _run_dir(tmp_path):
    return tmp_path / "ontology-validation" / RUN


def test_the_run_publishes_every_artifact(tmp_path) -> None:
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    out = _run_dir(tmp_path)
    for name in (REPORT_FILENAME, VIOLATIONS_FILENAME, SUMMARY_FILENAME, AXIOM_EVIDENCE_FILENAME):
        assert (out / name).exists(), name
    rows = [
        json.loads(line)
        for line in (out / VIOLATIONS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 9
    assert all(row["facts"] for row in rows)


def test_the_report_states_the_base_rate_per_axiom(tmp_path) -> None:
    _invoke(tmp_path)
    report = (_run_dir(tmp_path) / REPORT_FILENAME).read_text(encoding="utf-8")
    assert "| axiom | class |" in report
    assert "`func-diie`" in report
    assert "did not apply here" in report  # a zero is a finding, not a blank


def test_fail_on_violations_is_opt_in(tmp_path) -> None:
    assert _invoke(tmp_path).exit_code == 0
    assert _invoke(tmp_path, "--fail-on-violations").exit_code == 1


def test_an_unsigned_axiom_never_refuses_a_build(
    fixture_extractions: list[DocExtraction],
) -> None:
    check = check_build_inputs(fixture_extractions, CANDIDATE_TURTLE)
    assert check.signed_violations == []
    assert len(check.candidate_violations) == 9
    assert "reported, not refused" in " ".join(check.lines())


def test_a_signed_axiom_refuses_the_build(
    axiom_set: AxiomSet, fixture_extractions: list[DocExtraction], tmp_path
) -> None:
    signed = [
        axiom.model_copy(update={"signed_by": "reviewer", "signed_on": "2026-08-25"})
        if axiom.axiom_id == "func-diie"
        else axiom
        for axiom in axiom_set.axioms
    ]
    target = tmp_path / "signed.ttl"
    save_axioms(AxiomSet(version=axiom_set.version, axioms=signed), target, ["signed for the test"])
    check = check_build_inputs(fixture_extractions, target)
    assert [v.axiom_id for v in check.signed_violations] == ["func-diie"]
    assert "signed axiom broken" in " ".join(check.lines())


def test_build_graph_refuses_only_with_an_axiom_file(tmp_path) -> None:
    result = CliRunner().invoke(app, ["build-graph", "--refuse-violations"])
    assert result.exit_code == 2
    assert "--refuse-violations needs --axioms" in result.output
