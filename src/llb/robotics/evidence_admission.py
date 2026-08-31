"""Admission and evidence construction for HFlow text projections."""

from pathlib import Path

from llb.core.contracts.robotics import ProducerVersion, RoboticsEvidence
from llb.goldset.schema import load_goldset
from llb.goldset.verify_base import GOLDSET_FILENAME
from llb.goldset.verify_refcheck import check_verification_ref
from llb.robotics.evidence_models import AdmissionState, HflowProjectionRow
from llb.robotics.hflow_manifest import resolve_fixture_path


def _accepted_ledger_path(path: Path) -> Path:
    return path / GOLDSET_FILENAME if path.is_dir() else path


def _validate_model_verification(root: Path, row: HflowProjectionRow, text: str) -> None:
    if row.verification_ref is None:
        raise ValueError(
            f"projection {row.projection_id}: model-authored verified text needs verification_ref"
        )
    ref_path = resolve_fixture_path(root, row.verification_ref)
    status = check_verification_ref(ref_path)
    if not status.valid or status.kind != "accepted_ledger":
        raise ValueError(
            f"projection {row.projection_id}: verification reference is not an accepted ledger"
        )
    items = [
        item
        for item in load_goldset(_accepted_ledger_path(ref_path))
        if item.id == row.projection_id
    ]
    if len(items) != 1:
        raise ValueError(
            f"projection {row.projection_id}: accepted ledger must contain exactly one matching item"
        )
    matching_spans = [
        span
        for span in items[0].source_spans
        if span.doc_id == row.projection_uri
        and span.char_start == row.projection_start
        and span.char_end == row.projection_end
        and span.text == text
    ]
    if len(matching_spans) != 1:
        raise ValueError(
            f"projection {row.projection_id}: accepted ledger does not bind the exact source span"
        )
    corpus_path = ref_path.parent / "corpus" / row.projection_uri
    corpus_text = corpus_path.read_text(encoding="utf-8") if corpus_path.is_file() else ""
    if corpus_text[row.projection_start : row.projection_end] != text:
        raise ValueError(
            f"projection {row.projection_id}: accepted ledger corpus does not ground the span"
        )


def projection_admission(
    root: Path, row: HflowProjectionRow, text: str
) -> tuple[AdmissionState, str]:
    if row.quality_state == "quarantined":
        return "quarantined", "HFlow quality state is quarantined"
    if row.quality_state == "unverified":
        return "unverified", "HFlow critical checks did not produce a settled verdict"
    if not row.verified:
        return "draft", "projection has not passed its verification gate"
    if row.authored_by == "model":
        _validate_model_verification(root, row, text)
    return "accepted", "quality and projection verification gates passed"


def projection_evidence(row: HflowProjectionRow) -> RoboticsEvidence:
    producer_versions = (
        ProducerVersion(
            producer="hflow",
            version=f"{row.hflow_release}@{row.hflow_revision}",
        ),
        ProducerVersion(producer="hflow-schema", version=row.hflow_schema_version),
        ProducerVersion(producer="hflow-pipeline", version=row.pipeline_version),
        ProducerVersion(producer="hflow-curation-query", version=row.curation_query_digest),
        *row.check_versions,
        *row.enrichment_versions,
    )
    return RoboticsEvidence(
        schema_version=1,
        evidence_id=f"hflow:{row.episode_id}:{row.projection_id}",
        episode_id=row.episode_id,
        mcap_uri=row.mcap_uri,
        mcap_sha256=row.mcap_sha256,
        channels=row.channels,
        start_ns=row.start_ns,
        end_ns=row.end_ns,
        producer_versions=producer_versions,
        quality_state=row.quality_state,
        projection_uri=row.projection_uri,
        projection_start=row.projection_start,
        projection_end=row.projection_end,
    )
