"""Project a pinned HFlow Parquet manifest into the existing text-corpus boundary."""

import logging
from collections import Counter
from pathlib import Path
from typing import cast

from llb.bench.common import new_run_timestamp
from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.core.fsutil import atomic_write_text
from llb.core.paths import resolve_data_dir
from llb.prep.corpus.fingerprints import corpus_fingerprint
from llb.prep.corpus.ingest import ingest_corpus
from llb.rag.retrieval import chunk_hits_span
from llb.robotics.digests import file_digest
from llb.robotics.evidence_admission import projection_admission, projection_evidence
from llb.robotics.evidence_models import (
    CorpusSpan,
    EvidenceLedgerEntry,
    HflowGeneration,
    HflowProjectionRow,
)
from llb.robotics.evidence_report import write_evidence_report
from llb.robotics.hflow_manifest import (
    load_fixture_manifest,
    load_projection_manifest,
    resolve_fixture_path,
)
from llb.robotics.mcap_validation import validate_mcap_window
from llb.robotics.upstreams import HFLOW_RELEASE, HFLOW_REVISION

LOGGER = logging.getLogger(__name__)
METHOD_NAME = "robotics-evidence"
EVIDENCE_LEDGER_NAME = "evidence-ledger.jsonl"


def _validate_generation(rows: tuple[HflowProjectionRow, ...]) -> HflowGeneration:
    generation = rows[0].generation()
    if generation.hflow_release != HFLOW_RELEASE or generation.hflow_revision != HFLOW_REVISION:
        raise ValueError(
            f"projection manifest must pin {HFLOW_RELEASE}@{HFLOW_REVISION}, got "
            f"{generation.hflow_release}@{generation.hflow_revision}"
        )
    for row in rows[1:]:
        if row.generation() != generation:
            raise ValueError(f"projection {row.projection_id}: mixed HFlow generation")
    for label, versions in (
        ("check", generation.check_versions),
        ("enrichment", generation.enrichment_versions),
    ):
        producers = [version.producer for version in versions]
        if len(producers) != len(set(producers)):
            raise ValueError(f"projection manifest contains duplicate {label} producers")
    return generation


def _projection_text(root: Path, row: HflowProjectionRow) -> str:
    path = resolve_fixture_path(root, row.projection_uri)
    observed = file_digest(path)
    if observed != row.projection_sha256:
        raise ValueError(
            f"projection {row.projection_id}: expected {row.projection_sha256}, observed {observed}"
        )
    text = path.read_text(encoding="utf-8")
    if row.projection_end > len(text):
        raise ValueError(f"projection {row.projection_id}: character interval is out of range")
    projected = text[row.projection_start : row.projection_end]
    if not projected.strip():
        raise ValueError(f"projection {row.projection_id}: character interval is empty")
    return projected


def _validate_retrieval_span(span: CorpusSpan) -> None:
    source_span = cast(SourceSpanRecord, span.model_dump())
    chunk = cast(ChunkRecord, span.model_dump())
    if not chunk_hits_span(chunk, source_span):
        raise ValueError(f"{span.doc_id}: staged source span does not survive retrieval validation")


def run_evidence_bridge(
    fixture_root: Path,
    *,
    data_dir: Path | str | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Validate a pinned HFlow manifest and stage only headline-admitted text."""
    root = Path(fixture_root).resolve()
    fixture = load_fixture_manifest(root)
    manifest_path = resolve_fixture_path(root, fixture.manifest)
    rows = load_projection_manifest(manifest_path)
    generation = _validate_generation(rows)
    if output_dir is None:
        _run_id, run_timestamp = new_run_timestamp()
        output_dir = resolve_data_dir(data_dir) / METHOD_NAME / run_timestamp
    output_dir = Path(output_dir)
    staging_root = output_dir / "staging"
    corpus_root = output_dir / "corpus"

    entries: list[EvidenceLedgerEntry] = []
    inspected_episodes: set[str] = set()
    validated_windows: set[tuple[object, ...]] = set()
    for row in rows:
        text = _projection_text(root, row)
        mcap_path = resolve_fixture_path(root, row.mcap_uri)
        window_key = (
            row.mcap_uri,
            row.mcap_sha256,
            row.episode_id,
            row.channels,
            row.start_ns,
            row.end_ns,
        )
        if window_key not in validated_windows:
            validate_mcap_window(
                mcap_path,
                expected_sha256=row.mcap_sha256,
                episode_id=row.episode_id,
                channels=row.channels,
                start_ns=row.start_ns,
                end_ns=row.end_ns,
            )
            validated_windows.add(window_key)
        inspected_episodes.add(row.episode_id)
        admission, reason = projection_admission(root, row, text)
        source_span: CorpusSpan | None = None
        if admission == "accepted":
            doc_id = f"robotics/{row.episode_id}/{row.projection_id}.md"
            atomic_write_text(staging_root / doc_id, text)
            source_span = CorpusSpan(
                doc_id=doc_id,
                char_start=0,
                char_end=len(text),
                text=text,
            )
            _validate_retrieval_span(source_span)
        entries.append(
            EvidenceLedgerEntry(
                schema_version=1,
                projection_id=row.projection_id,
                admission=admission,
                admission_reason=reason,
                projection_kind=row.projection_kind,
                authored_by=row.authored_by,
                language=row.language,
                verification_ref=row.verification_ref,
                quarantine_tags=row.quarantine_tags,
                generation=generation,
                source_span=source_span,
                evidence=projection_evidence(row),
            )
        )

    if not any(entry.admission == "accepted" for entry in entries):
        raise ValueError("HFlow manifest has no projection admitted to the corpus")
    ingest_result = ingest_corpus(
        staging_root,
        corpus_root,
        min_chars=1,
        default_language="und",
        source_system="hflow",
    )
    ledger_path = output_dir / EVIDENCE_LEDGER_NAME
    atomic_write_text(
        ledger_path,
        "".join(entry.model_dump_json() + "\n" for entry in entries),
    )
    counts = Counter(entry.admission for entry in entries)
    report: dict[str, object] = {
        "schema_version": 1,
        "verdict": "pass",
        "projection_count": len(entries),
        "episode_count": len(inspected_episodes),
        "admission_counts": dict(sorted(counts.items())),
        "corpus_documents": ingest_result.n_docs,
        "corpus_fingerprint": corpus_fingerprint(corpus_root),
        "evidence_ledger_sha256": file_digest(ledger_path),
        "projection_manifest_sha256": file_digest(manifest_path),
        "generation": generation.model_dump(mode="json"),
        "source_span_validation": "pass",
        "standard_mcap_validation": "pass",
    }
    write_evidence_report(output_dir, report)
    LOGGER.info("HFlow robotics evidence bridge report written to %s", output_dir)
    return output_dir, report
