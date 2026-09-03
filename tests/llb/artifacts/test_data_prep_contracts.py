"""The data-preparation exchange surface reads through the registry, at every form it was written.

The fixtures under `samples/artifact_contracts/data_prep/` are the two shapes that matter: a
bundle written by this build, and the pre-contract form the same bundle had before the registry
existed. Every assertion here is that the two reach the same domain values, or that a form this
build cannot read is refused where an operator can still act on the refusal.
"""

import json
import shutil
from pathlib import Path

import pytest

from llb.artifacts.bundles import corpus_bundle_manifest, draft_bundle_manifest
from llb.artifacts.dataset_reading import read_dataset
from llb.artifacts.data_prep.families import data_prep_definitions
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import DatasetReadError, UnsupportedFutureVersionError
from llb.artifacts.gates import (
    ArtifactCompatibilityError,
    refuse_unreadable_corpus,
    refuse_unreadable_review,
)
from llb.conflicts.bundle.contract import stage_inputs_at_current
from llb.conflicts.bundle.readings import bundle_readings
from llb.conflicts.bundle.record import documents_of, recorded_inputs
from llb.conflicts.constants import COVERAGE_FIELD, STAGE_INPUTS_FIELD
from llb.conflicts.resolution.overlay import overlay_from_plan
from llb.goldset.schema import load_goldset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "samples" / "artifact_contracts" / "data_prep"
CORPUS_BUNDLE = FIXTURES / "corpus"
DRAFT_BUNDLE = FIXTURES / "draft-bundle"
LEGACY = FIXTURES / "legacy"
CURRENT = FIXTURES / "current"


def _read(path: Path, schema_id: str) -> dict[str, object]:
    record = DEFAULT_REGISTRY.read_as(
        schema_id, json.loads(path.read_text(encoding="utf-8")), source=str(path)
    )
    return record.model_dump()


def test_corpus_bundle_validates_member_by_member() -> None:
    manifest = corpus_bundle_manifest(CORPUS_BUNDLE)
    members = read_dataset(CORPUS_BUNDLE, manifest)

    assert [member.member_id for member in manifest.members] == [
        "corpus-manifest",
        "conflict-overlay",
    ]
    assert members["corpus-manifest"][0].model_dump()["items"][0]["doc_id"] == "poryadok.md"
    assert members["conflict-overlay"][0].model_dump()["policy"] == "prefer-newer"


def test_draft_bundle_validates_member_by_member() -> None:
    manifest = draft_bundle_manifest(DRAFT_BUNDLE)
    members = read_dataset(DRAFT_BUNDLE, manifest)

    assert [member.member_id for member in manifest.members] == [
        "gold-items",
        "gold-chains",
        "ontology",
        "extraction",
        "provenance",
    ]
    assert all(records for records in members.values())


def test_a_tampered_member_refuses_before_the_bundle_is_read(tmp_path: Path) -> None:
    bundle = tmp_path / "draft"
    shutil.copytree(DRAFT_BUNDLE, bundle)
    manifest = draft_bundle_manifest(bundle)
    (bundle / "ontology.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DatasetReadError, match="digest mismatch"):
        read_dataset(bundle, manifest)


def test_a_pre_contract_gold_set_reaches_the_same_items() -> None:
    legacy = [item.model_dump() for item in load_goldset(LEGACY / "goldset.jsonl")]
    current = [item.model_dump() for item in load_goldset(DRAFT_BUNDLE / "goldset.jsonl")]

    assert legacy == current
    assert current[0]["schema_version"] == "2.0.0"


def test_a_pre_binding_provenance_states_the_absence_it_could_not_record() -> None:
    legacy = _read(LEGACY / "provenance.json", "llb.ontology-provenance")
    current = _read(DRAFT_BUNDLE / "provenance.json", "llb.ontology-provenance")

    assert legacy["corpus_version"] is None and current["corpus_version"] is not None
    assert {key: value for key, value in legacy.items() if key != "corpus_version"} == {
        key: value for key, value in current.items() if key != "corpus_version"
    }


def test_a_pre_contract_linkage_bundle_reaches_the_run_settings_it_was_written_under() -> None:
    legacy = _read(LEGACY / "linkage-settings.json", "llb.linkage-settings")
    current = _read(CURRENT / "linkage-settings.json", "llb.linkage-settings")

    assert legacy == current
    assert legacy["specification"]["match_threshold"] == 0.9


def test_an_early_conflict_record_replays_the_same_readings() -> None:
    legacy = json.loads((LEGACY / "conflict-stage-inputs.json").read_text(encoding="utf-8"))
    current = json.loads((CURRENT / "conflict-stage-inputs.json").read_text(encoding="utf-8"))
    migrated = stage_inputs_at_current(legacy, source=LEGACY)

    assert documents_of(migrated) == documents_of(current)
    assert recorded_inputs(migrated) == recorded_inputs(current)
    assert _readings(migrated) == _readings(current)


def test_an_unresolvable_conflict_record_refuses_rather_than_reading_as_empty() -> None:
    record = json.loads((CURRENT / "conflict-stage-inputs.json").read_text(encoding="utf-8"))
    record["schema_version"] = 6
    record["chunks"] = "not an accounting"

    readings = bundle_readings(_summary(record))

    assert all(not reading.available for reading in readings)
    assert any("source record is invalid" in reading.detail for reading in readings)


def _summary(record: dict[str, object]) -> dict[str, object]:
    return {STAGE_INPUTS_FIELD: record, COVERAGE_FIELD: {"orderable_pairs": 1}}


def _readings(record: dict[str, object]) -> list[tuple[str, bool]]:
    return [(reading.name, reading.available) for reading in bundle_readings(_summary(record))]


def test_a_future_major_refuses_before_a_store_build(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS_BUNDLE, corpus)
    shutil.copy(
        FIXTURES / "unsupported-future" / "corpus_manifest.json", corpus / "corpus_manifest.json"
    )

    with pytest.raises(ArtifactCompatibilityError, match="version is not supported"):
        refuse_unreadable_corpus(corpus)


def test_a_future_major_refuses_before_a_review_session_opens(tmp_path: Path) -> None:
    bundle = tmp_path / "draft"
    shutil.copytree(DRAFT_BUNDLE, bundle)
    record = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    record["schema_version"] = "3.0.0"
    (bundle / "provenance.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError):
        refuse_unreadable_review(bundle / "worksheet.csv")


def test_the_future_major_is_named_as_such_by_the_registry() -> None:
    record = json.loads(
        (FIXTURES / "unsupported-future" / "corpus_manifest.json").read_text(encoding="utf-8")
    )

    with pytest.raises(UnsupportedFutureVersionError):
        DEFAULT_REGISTRY.read_current(record, source="unsupported-future")


def test_every_data_prep_family_declares_how_its_own_history_is_read() -> None:
    for definition in data_prep_definitions():
        assert definition.legacy_version in definition.models, definition.schema_id


def test_the_overlay_producer_writes_the_form_a_fingerprint_hashes() -> None:
    overlay = overlay_from_plan(
        {
            "policy": "prefer-newer",
            "source_findings_sha256": "0" * 64,
            "items": [],
        }
    )

    assert overlay["schema_version"] == 1 and "schema_id" not in overlay
    assert DEFAULT_REGISTRY.read_as("llb.conflict-overlay", overlay, version="1.0.0")


def test_a_freshly_ingested_corpus_writes_its_manifest_at_the_current_contract(
    tmp_path: Path,
) -> None:
    from llb.prep.corpus.ingest import ingest_corpus

    source = tmp_path / "src"
    source.mkdir()
    (source / "poryadok.md").write_text((CORPUS_BUNDLE / "poryadok.md").read_text("utf-8") * 8)
    staged = tmp_path / "staged"

    ingest_corpus(source, staged, min_chars=50)

    manifest = _read(staged / "corpus_manifest.json", "llb.corpus-manifest")
    assert manifest["schema_version"] == "1.0.0" and manifest["kind"] == "corpus"
    refuse_unreadable_corpus(staged)


def test_check_bundle_upgrades_a_pre_contract_bundle_in_place(tmp_path: Path) -> None:
    from llb.artifacts.bundles import draft_bundle_manifest
    from llb.artifacts.dataset_reading import survey_dataset, upgrade_dataset

    bundle = tmp_path / "draft"
    shutil.copytree(DRAFT_BUNDLE, bundle)
    shutil.copy(LEGACY / "goldset.jsonl", bundle / "goldset.jsonl")
    shutil.copy(LEGACY / "provenance.json", bundle / "provenance.json")

    upgraded = upgrade_dataset(bundle, draft_bundle_manifest(bundle))

    assert set(upgraded) == {"gold-items", "provenance"}
    readings = survey_dataset(bundle, draft_bundle_manifest(bundle))
    assert all(not reading.needs_upgrade and not reading.refusal for reading in readings)
    assert upgrade_dataset(bundle, draft_bundle_manifest(bundle)) == ()


def test_survey_reports_every_refusal_rather_than_the_first(tmp_path: Path) -> None:
    from llb.artifacts.bundles import draft_bundle_manifest
    from llb.artifacts.dataset_reading import survey_dataset

    bundle = tmp_path / "draft"
    shutil.copytree(DRAFT_BUNDLE, bundle)
    manifest = draft_bundle_manifest(bundle)
    (bundle / "ontology.json").write_text("{}", encoding="utf-8")
    (bundle / "chains.jsonl").write_text("{}\n", encoding="utf-8")

    readings = survey_dataset(bundle, manifest)

    assert {r.member_id for r in readings if r.refusal} == {"ontology", "gold-chains"}
