"""The run-bundle, board, and orchestration contracts, read against committed fixtures.

`samples/artifact_contracts/run_bundles/` holds an evaluation bundle and a benchmark bundle at the
current contracts, the pre-contract form of each under `legacy/`, and a manifest from an
unsupported future major. The question every test here asks is the migration's own: does a bundle
this project wrote before the families existed read to the SAME logical records as one written
after, and is a bundle this build cannot read named rather than aggregated.
"""

import json
import shutil
from pathlib import Path

import pytest

from llb.artifacts.datasets import load_dataset_manifest
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import (
    ArtifactContractError,
    DatasetReadError,
    UnsupportedFutureVersionError,
)
from llb.artifacts.gates import ArtifactCompatibilityError, refuse_unreadable_run_bundle
from llb.artifacts.runs.bundle import (
    read_case_rows,
    read_case_series,
    read_retrieval_rows,
    read_run_manifest,
    read_score_rows,
    read_study_analysis,
    read_study_design,
    run_bundle_kind,
)
from llb.artifacts.runs.datasets import KIND_BENCHMARK, KIND_RUN, run_bundle_manifest
from llb.artifacts.runs.fixture import case_score_row, run_metrics
from llb.artifacts.runs.members import (
    HUMAN_REPORT,
    RunMember,
    human_report,
    member_problems,
    study_analysis,
    study_design,
)
from llb.artifacts.runs.rows import decode_record, read_rows
from llb.board.runs import load_run_records
from llb.core.contracts.orchestration import (
    AGENT_PROFILE_SCHEMA_ID,
    AUTO_RAG_JOURNAL_EVENT_SCHEMA_ID,
    AUTO_RAG_MANIFEST_SCHEMA_ID,
    AUTO_RAG_RECOMMENDATION_SCHEMA_ID,
    AUTO_RAG_STAGE_LINKS_SCHEMA_ID,
    AUTO_RAG_STAGE_RESULT_SCHEMA_ID,
    MISS_ANALYSIS_SCHEMA_ID,
    MISS_RECORD_SCHEMA_ID,
)
from llb.core.contracts.run_bundle import (
    BENCHMARK_CELL_SCHEMA_ID,
    CASE_PROGRESS_SCHEMA_ID,
    CASE_RETRIEVAL_SCHEMA_ID,
    CASE_SCORE_SCHEMA_ID,
    CONTEXT_PROBE_SCHEMA_ID,
    RUN_ABORT_SCHEMA_ID,
    RUN_PROGRESS_META_SCHEMA_ID,
    STUDY_ANALYSIS_SCHEMA_ID,
    STUDY_DESIGN_SCHEMA_ID,
)
from llb.core.contracts.runs import RUN_MANIFEST_SCHEMA_ID, RunManifest
from llb.tracking.manifest import persist_run

FIXTURES = Path(__file__).resolve().parents[3] / "samples" / "artifact_contracts" / "run_bundles"
CURRENT_RUN = FIXTURES / "run"
CURRENT_BENCHMARK = FIXTURES / "benchmark"
LEGACY_RUN = FIXTURES / "legacy" / "run"
LEGACY_BENCHMARK = FIXTURES / "legacy" / "benchmark"
UNSUPPORTED = FIXTURES / "unsupported-future"

RUN_FAMILIES = (
    RUN_MANIFEST_SCHEMA_ID,
    CASE_SCORE_SCHEMA_ID,
    BENCHMARK_CELL_SCHEMA_ID,
    CASE_RETRIEVAL_SCHEMA_ID,
    CASE_PROGRESS_SCHEMA_ID,
    RUN_PROGRESS_META_SCHEMA_ID,
    RUN_ABORT_SCHEMA_ID,
    CONTEXT_PROBE_SCHEMA_ID,
    STUDY_DESIGN_SCHEMA_ID,
    STUDY_ANALYSIS_SCHEMA_ID,
    MISS_ANALYSIS_SCHEMA_ID,
    MISS_RECORD_SCHEMA_ID,
    AGENT_PROFILE_SCHEMA_ID,
    AUTO_RAG_MANIFEST_SCHEMA_ID,
    AUTO_RAG_JOURNAL_EVENT_SCHEMA_ID,
    AUTO_RAG_STAGE_RESULT_SCHEMA_ID,
    AUTO_RAG_STAGE_LINKS_SCHEMA_ID,
    AUTO_RAG_RECOMMENDATION_SCHEMA_ID,
)


def _manifest(**overrides: object) -> RunManifest:
    fields: dict[str, object] = {
        "run_id": "run-1",
        "run_name": "fixture",
        "split": "final",
        "config": {"model": "m"},
        "metrics": run_metrics(objective_score=0.5),
        "n_cases": 1,
    }
    fields.update(overrides)
    return RunManifest(**fields)  # type: ignore[arg-type]


def test_a_pre_contract_bundle_reads_to_the_same_records_as_the_current_one():
    assert read_run_manifest(LEGACY_RUN) == read_run_manifest(CURRENT_RUN)
    assert read_score_rows(LEGACY_RUN) == read_score_rows(CURRENT_RUN)
    assert read_retrieval_rows(LEGACY_RUN) == read_retrieval_rows(CURRENT_RUN)
    assert read_rows(LEGACY_RUN / "probes.jsonl") == read_rows(CURRENT_RUN / "probes.jsonl")
    assert read_study_design(LEGACY_RUN / "second-fold-design.json") == read_study_design(
        CURRENT_RUN / "second-fold-design.json"
    )
    assert read_study_analysis(LEGACY_RUN / "second-fold-analysis.json") == read_study_analysis(
        CURRENT_RUN / "second-fold-analysis.json"
    )


def test_a_pre_contract_bundle_states_its_own_kind(tmp_path):
    """A benchmark bundle records the category it is a cell of; an evaluation bundle has none."""
    for source, expected in ((LEGACY_RUN, KIND_RUN), (LEGACY_BENCHMARK, KIND_BENCHMARK)):
        bundle = tmp_path / expected
        shutil.copytree(source, bundle)
        assert run_bundle_kind(bundle) == expected
        refuse_unreadable_run_bundle(bundle)


def test_a_benchmark_cell_reads_flat_in_both_forms():
    """The envelope is on disk; every lane that reads a cell sees the columns it wrote."""
    current = read_score_rows(CURRENT_BENCHMARK)
    assert current == read_score_rows(LEGACY_BENCHMARK)
    assert current[0]["cell_id"] == "surface-d6" and "cell" not in current[0]
    assert read_case_series(CURRENT_BENCHMARK, "n_steps") == [6.0, 11.0]


def test_a_loaded_record_never_carries_its_identity():
    for row in read_score_rows(CURRENT_RUN) + read_retrieval_rows(CURRENT_RUN):
        assert "schema_id" not in row and "schema_version" not in row
    assert "schema_id" not in read_run_manifest(CURRENT_RUN)


def test_every_run_family_is_at_one_initial_version_with_no_migration():
    for schema_id in RUN_FAMILIES:
        definition = DEFAULT_REGISTRY.definition(schema_id)
        assert definition.current_version == "1.0.0"
        assert definition.supported_versions == ("1.0.0",)
        assert definition.migrations == () and definition.refusals == ()
        assert definition.legacy_version == "1.0.0"


def test_a_case_row_missing_a_required_column_never_reaches_disk(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ArtifactContractError, match="source record is invalid"):
        persist_run(_manifest(), [{"item_id": "x"}], out, mirror=lambda *_a: None)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_publication_refuses_a_staged_member_it_cannot_read_back(tmp_path):
    """The rename is behind the read-back, so a half-readable bundle is never visible."""
    out = tmp_path / "run"
    staging = tmp_path / ".run.tmp"
    staging.mkdir()
    (staging / "scorer").mkdir()
    (staging / "scorer" / "abort.json").write_text(
        json.dumps({"status": "aborted"}), encoding="utf-8"
    )
    with pytest.raises(DatasetReadError, match="cannot be read back"):
        persist_run(
            _manifest(),
            [case_score_row("x")],
            out,
            mirror=lambda *_a: None,
            staging_dir=staging,
        )
    assert not out.exists() and not staging.exists()


def test_a_published_bundle_describes_itself_including_its_score_contract(tmp_path):
    run, benchmark = tmp_path / "run", tmp_path / "bench"
    persist_run(_manifest(), [case_score_row("x")], run, mirror=lambda *_a: None)
    persist_run(
        _manifest(config={"category": "agentic"}),
        [{"cell_id": "c1", "success": 1.0}],
        benchmark,
        mirror=lambda *_a: None,
        kind=KIND_BENCHMARK,
    )
    for root, expected in ((run, CASE_SCORE_SCHEMA_ID), (benchmark, BENCHMARK_CELL_SCHEMA_ID)):
        described = load_dataset_manifest(root)
        assert described is not None
        scores = next(member for member in described.members if member.member_id == "scores")
        assert scores.record_contract is not None
        assert scores.record_contract.schema_id == expected


def test_an_additional_member_needs_a_contract_or_the_human_report_exemption():
    ok = [
        study_design("d.json", {"study_id": "s"}),
        study_analysis("a.json", {"reading": "collapses"}),
        human_report("r.md", "# report\n"),
    ]
    assert member_problems(ok) == ()
    problems = member_problems(
        [
            RunMember("raw.json", "{}", "llb.not-a-family"),
            RunMember("notes.json", "{}", HUMAN_REPORT),
            RunMember("../escape.md", "x", HUMAN_REPORT),
        ]
    )
    assert any("unregistered contract" in problem for problem in problems)
    assert any("human-report exemption covers Markdown only" in problem for problem in problems)
    assert any("plain file name" in problem for problem in problems)


def test_an_unregistered_member_is_refused_before_anything_is_published(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="unregistered contract"):
        persist_run(
            _manifest(),
            [case_score_row("x")],
            out,
            mirror=lambda *_a: None,
            artifacts=[RunMember("raw.json", "{}", "llb.not-a-family")],
        )
    assert not out.exists()


def test_a_future_major_refuses_at_the_reader_and_at_the_board(tmp_path):
    with pytest.raises(UnsupportedFutureVersionError):
        read_run_manifest(UNSUPPORTED / "manifest.json")

    root = tmp_path / "run-eval"
    shutil.copytree(UNSUPPORTED, root / "20260901T000000Z-future")
    with pytest.raises(UnsupportedFutureVersionError):
        load_run_records(root)


def test_the_bundle_gate_reports_every_unreadable_member_at_once(tmp_path):
    bundle = tmp_path / "run"
    shutil.copytree(CURRENT_RUN, bundle)
    refuse_unreadable_run_bundle(bundle)

    (bundle / "scores.jsonl").write_text(
        json.dumps({"schema_id": CASE_SCORE_SCHEMA_ID, "schema_version": "9.0.0"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactCompatibilityError, match="run bundle cannot be read"):
        refuse_unreadable_run_bundle(bundle)


def test_a_bundle_published_before_descriptions_existed_still_reads(tmp_path):
    """No `dataset_manifest.json` is not a refusal -- every bundle written before it is one."""
    bundle = tmp_path / "legacy-run"
    shutil.copytree(LEGACY_RUN, bundle)
    assert load_dataset_manifest(bundle) is None
    refuse_unreadable_run_bundle(bundle)
    described = run_bundle_manifest(bundle, kind=KIND_RUN)
    assert {member.member_id for member in described.members} >= {
        "run-manifest",
        "scores",
        "retrieval",
        "probes",
        "budget-abort",
    }


def test_a_row_naming_another_family_is_refused_by_the_reader(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps({"schema_id": BENCHMARK_CELL_SCHEMA_ID, "schema_version": "1.0.0", "cell": {}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetReadError, match="expected"):
        read_rows(path, CASE_SCORE_SCHEMA_ID)


def test_per_case_rows_refuse_an_artifact_of_another_shape(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({"objective_score": 1.0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="per-case score row"):
        read_case_rows(path)


def test_the_journal_and_the_resume_meta_read_back_in_both_forms():
    """Both are staging-only, so their pre-contract form is what an interrupted run left."""
    legacy_progress = {"item_id": "case-1", "state": {"answer": "Kyiv", "status": "ok"}}
    current = decode_record(
        CASE_PROGRESS_SCHEMA_ID,
        {"schema_id": CASE_PROGRESS_SCHEMA_ID, "schema_version": "1.0.0", **legacy_progress},
    )
    assert current == decode_record(CASE_PROGRESS_SCHEMA_ID, legacy_progress) == legacy_progress

    legacy_meta = {
        "run_id": "r",
        "split": "final",
        "config_digest": "a" * 8,
        "goldset_digest": "b" * 8,
        "n_items": 3,
    }
    assert decode_record(RUN_PROGRESS_META_SCHEMA_ID, legacy_meta) == legacy_meta
