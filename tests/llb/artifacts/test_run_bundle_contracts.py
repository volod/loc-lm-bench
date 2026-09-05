"""Run bundles, study records, and orchestration trails read through their contracts.

The two bundles under `samples/artifact_contracts/run_bundles/` are the same run written twice:
`current/` as this build publishes it, `legacy/` as this project wrote it before the registry
existed. Every test below asks one of two questions -- does the old form come back as the record a
current writer would produce, and does a form this build cannot read refuse before a board, a
study, or an external consumer acts on it.
"""

import json
from pathlib import Path

import pytest

from llb.artifacts.errors import (
    ArtifactContractError,
    DatasetReadError,
    MissingIdentityError,
    UnsupportedFutureVersionError,
)
from llb.artifacts.run_bundle.datasets import run_bundle_manifest
from llb.artifacts.run_bundle.journals import read_budget_abort, read_progress_meta
from llb.artifacts.run_bundle.manifests import declare_score_rows, read_run_manifest
from llb.artifacts.run_bundle.run_artifacts import human_report, study_artifact
from llb.artifacts.run_bundle.studies import read_study_record, study_record
from llb.artifacts.run_bundle.survey import survey_run_bundle
from llb.bench.artifacts import declared_artifacts
from llb.board.io import admitted_manifest
from llb.board.runs import load_run_records
from llb.core.contracts.run_bundle.journals import CaseProgressMeta, JudgeBudgetAbort
from llb.core.contracts.run_bundle.rows import CASE_SCORE_SCHEMA_ID
from llb.core.contracts.run_bundle.studies import (
    STUDY_ANALYSIS_SCHEMA_ID,
    STUDY_DESIGN_SCHEMA_ID,
)
from llb.tracking.manifest import RunManifest, persist_run

BUNDLES = Path("samples/artifact_contracts/run_bundles")
CURRENT = BUNDLES / "current"
LEGACY = BUNDLES / "legacy"


def _survey(root: Path) -> dict[str, str]:
    """Every member of one bundle, mapped to its refusal (empty when it read back)."""
    return {
        reading.member_id: reading.refusal
        for reading in survey_run_bundle(root, run_bundle_manifest(root))
    }


def test_current_bundle_reads_every_declared_member():
    readings = survey_run_bundle(CURRENT, run_bundle_manifest(CURRENT))
    by_id = {reading.member_id: reading for reading in readings}
    assert not [reading for reading in readings if reading.refusal]
    assert by_id["scores"].records == 2
    assert by_id["retrieval"].records == 2
    # The Markdown report is declared exempt, so it is bound by its bytes rather than a contract.
    assert by_id["artifact-2"].schema_id == ""


def test_pre_contract_bundle_reads_as_the_record_a_current_writer_produces():
    old = read_run_manifest(LEGACY / "manifest.json")
    new = read_run_manifest(CURRENT / "manifest.json")
    assert old.schema_version == "2.0.0"
    # The two absences are exactly what an older bundle could not record: what its rows answered
    # to, and which additional files it published. Everything else is the same run.
    assert old.score_rows is None and old.artifacts == []
    assert old.model_dump(exclude={"score_rows", "artifacts"}) == new.model_dump(
        exclude={"score_rows", "artifacts"}
    )
    assert not [refusal for refusal in _survey(LEGACY).values() if refusal]


def test_supported_old_bundle_yields_the_same_board_reading(tmp_path):
    """The gate an archived run depends on: migrating it must not move its published numbers."""
    readings = {}
    for name, source in (("current", CURRENT), ("legacy", LEGACY)):
        root = tmp_path / name / "run-eval" / "2026-01-01T00-00-00Z"
        root.mkdir(parents=True)
        for member in source.iterdir():
            (root / member.name).write_bytes(member.read_bytes())
        records = load_run_records(root.parent)
        assert len(records) == 1
        readings[name] = (
            records[0].result.objective_score,
            records[0].result.n_cases,
            records[0].result.case_objectives,
            records[0].result.semantic_score,
        )
    assert readings["current"] == readings["legacy"]


def test_future_major_manifest_refuses_before_anything_reads_it():
    with pytest.raises(UnsupportedFutureVersionError):
        read_run_manifest(BUNDLES / "unsupported-future" / "manifest.json")
    assert admitted_manifest(BUNDLES / "unsupported-future" / "manifest.json") is None


def test_mixed_version_rows_refuse_before_board_admission():
    mixed = BUNDLES / "mixed-version"
    assert admitted_manifest(mixed / "manifest.json") is None
    assert "llb.agentic-case" in _survey(mixed)["scores"]


def test_score_rows_are_validated_before_the_bundle_is_published(tmp_path):
    manifest = RunManifest(run_id="r1", run_name="t", config={"model": "m"}, n_cases=1)
    with pytest.raises(ArtifactContractError):
        persist_run(
            manifest,
            [{"item_id": "case-1", "objective_score": "not a number"}],
            tmp_path / "run",
            mirror=lambda *args: None,
            score_contract=CASE_SCORE_SCHEMA_ID,
        )
    assert not (tmp_path / "run").exists()


def test_a_bundle_must_say_what_its_rows_answer_to(tmp_path):
    manifest = RunManifest(run_id="r1", run_name="t", config={"model": "m"}, n_cases=1)
    with pytest.raises(ValueError, match="registered contract or an owning study"):
        persist_run(manifest, [{"item_id": "x"}], tmp_path / "run", mirror=lambda *args: None)


def test_study_rows_publish_their_own_column_set(tmp_path):
    manifest = RunManifest(run_id="r1", run_name="t", config={"model": "m"}, n_cases=1)
    out = tmp_path / "run"
    persist_run(
        manifest,
        [{"cell_id": "a", "completion_rate": 1.0}, {"cell_id": "b", "completion_rate": 0.5}],
        out,
        mirror=lambda *args: None,
        score_owner="agentic-loop-policy",
    )
    published = read_run_manifest(out / "manifest.json")
    assert published.score_rows is not None
    assert published.score_rows.owner == "agentic-loop-policy"
    assert published.score_rows.columns == ["cell_id", "completion_rate"]
    assert not [refusal for refusal in _survey(out).values() if refusal]


def test_a_row_carrying_an_undeclared_column_refuses(tmp_path):
    out = tmp_path / "run"
    persist_run(
        RunManifest(run_id="r1", run_name="t", config={"model": "m"}, n_cases=1),
        [{"cell_id": "a"}],
        out,
        mirror=lambda *args: None,
        score_owner="agentic-loop-policy",
    )
    scores = out / "scores.jsonl"
    scores.write_text(json.dumps({"cell_id": "a", "smuggled": 1}) + "\n", encoding="utf-8")
    assert "smuggled" in _survey(out)["scores"]


def test_an_artifact_is_either_a_contract_or_a_declared_human_report():
    design = (CURRENT / "study-design.json").read_text(encoding="utf-8")
    declared = declared_artifacts(
        {"x-design.json": design, "x.md": "# table\n"}, study_id="ignored"
    )
    kinds = {item.name: (item.record_contract, item.human_report) for item in declared}
    assert kinds["x-design.json"][0].schema_id == STUDY_DESIGN_SCHEMA_ID
    assert kinds["x.md"][1]
    with pytest.raises(DatasetReadError, match="neither"):
        declared_artifacts({"trace.bin": "\x00"}, study_id="s")


def test_a_study_reading_must_name_the_study_it_belongs_to():
    analysis = json.dumps({"activation_rate": 0.75})
    with pytest.raises(MissingIdentityError):
        study_artifact("a-analysis.json", STUDY_ANALYSIS_SCHEMA_ID, analysis, study_id="")
    named = study_artifact("a-analysis.json", STUDY_ANALYSIS_SCHEMA_ID, analysis, study_id="s-v1")
    assert named.declaration().study_id == "s-v1"


def test_a_study_record_round_trips_to_the_bytes_it_was_written_as():
    """A published aggregate is cited by digest, so the local form must not move."""
    path = CURRENT / "study-design.json"
    record = read_study_record(path, STUDY_DESIGN_SCHEMA_ID)
    assert record.study_id == "artifact-contract-sample-study-v1"
    assert record.local_version == 1
    assert json.dumps(record.body, indent=2, sort_keys=True) + "\n" == path.read_text(
        encoding="utf-8"
    )


def test_a_declared_artifact_whose_bytes_changed_refuses(tmp_path):
    out = tmp_path / "run"
    persist_run(
        RunManifest(run_id="r1", run_name="t", config={"model": "m"}, n_cases=1),
        [{"cell_id": "a"}],
        out,
        mirror=lambda *args: None,
        score_owner="study",
        artifacts=[human_report("comparison.md", "# before\n")],
    )
    (out / "comparison.md").write_text("# after\n", encoding="utf-8")
    assert "digest mismatch" in _survey(out)["artifact-1"]


def test_an_artifact_may_not_shadow_a_canonical_member(tmp_path):
    with pytest.raises(ValueError, match="may not be named"):
        persist_run(
            RunManifest(run_id="r1", run_name="t", config={"model": "m"}, n_cases=1),
            [{"cell_id": "a"}],
            tmp_path / "run",
            mirror=lambda *args: None,
            score_owner="study",
            artifacts=[human_report("scores.jsonl", "# not the score rows\n")],
        )
    assert not (tmp_path / "run").exists()


def test_a_resume_record_and_a_budget_abort_read_through_their_contracts(tmp_path):
    meta = CaseProgressMeta(
        run_id="r1", split="final", config_digest="c", goldset_digest="g", n_items=2
    )
    meta_path = tmp_path / "cases.progress.meta.json"
    meta_path.write_text(json.dumps(meta.model_dump(mode="json")), encoding="utf-8")
    assert read_progress_meta(meta_path) == meta

    abort = JudgeBudgetAbort(resumable=True, reason="max-usd", calls=3, cost_usd=1.5)
    abort_path = tmp_path / "abort.json"
    abort_path.write_text(json.dumps(abort.model_dump(mode="json")), encoding="utf-8")
    assert read_budget_abort(abort_path).resumable is True


def test_declaring_rows_refuses_a_shape_the_named_contract_does_not_describe():
    with pytest.raises(ArtifactContractError):
        declare_score_rows([{"item_id": "x"}], schema_id=CASE_SCORE_SCHEMA_ID)


def test_a_bare_study_table_is_still_attributable():
    """Three studies publish a schedule or a snapshot list rather than a record."""
    record = study_record([{"model_family": "gemma"}], STUDY_ANALYSIS_SCHEMA_ID, study_id="s-v1")
    assert record.study_id == "s-v1" and record.local_version is None
