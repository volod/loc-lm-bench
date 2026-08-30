"""Build the portable HFlow evidence fixture from real ``app.test()`` outputs."""

import json
import shutil
from pathlib import Path
from typing import Any

from llb.core.contracts.robotics import ProducerVersion
from llb.core.fsutil import atomic_write_text
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.robotics.digests import file_digest, value_digest
from llb.robotics.evidence_models import (
    HflowFixtureFile,
    HflowFixtureManifest,
    HflowProjectionRow,
)
from llb.robotics.hflow_manifest import (
    FIXTURE_MANIFEST_NAME,
    PROJECTION_MANIFEST_NAME,
    write_projection_manifest,
)
from llb.robotics.mcap_validation import inspect_mcap_channels
from llb.robotics.upstreams import HFLOW_RELEASE, HFLOW_REVISION

CHECK_NAME = "llb_bridge_quality"
CHECK_VERSION = "1"
ENRICHMENT_NAME = "llb_projection"
ENRICHMENT_VERSION = "1"
CURATION_SQL = """
SELECT episode_id, uri, schema_version, pipeline_version, status
FROM episodes
ORDER BY episode_id
""".strip()

_PROJECTIONS = (
    ("clean-human-label", "label", "human", True, "Clean joint-state episode."),
    (
        "clean-model-procedure",
        "procedure_summary",
        "model",
        True,
        "The robot follows a smooth joint-state trajectory.",
    ),
    (
        "clean-model-caption-draft",
        "caption",
        "model",
        False,
        "A draft caption describes smooth robot motion.",
    ),
    ("unverified-human-label", "label", "human", True, "Unsettled quality evidence."),
    ("quarantined-human-label", "label", "human", True, "Abrupt joint motion detected."),
)


def _copy_episode(root: Path, report: Any) -> tuple[str, str, str, tuple[str, ...], int, int]:
    if report.catalog_entry is None:
        raise ValueError("HFlow app.test(record=True) did not produce a catalog entry")
    episode_id = str(report.catalog_entry.episode_id)
    relative = f"episodes/{episode_id}.mcap"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report.canonical_path, destination)
    digest = file_digest(destination)
    channels = ("/joint_states",)
    window = inspect_mcap_channels(destination, channels)
    return (
        episode_id,
        relative,
        digest,
        channels,
        window.message_start_ns,
        window.message_end_ns,
    )


def _write_projection(root: Path, projection_id: str, text: str) -> tuple[str, str]:
    relative = f"projections/{projection_id}.md"
    atomic_write_text(root / relative, text)
    return relative, file_digest(root / relative)


def _row(
    *,
    report: Any,
    episode: tuple[str, str, str, tuple[str, ...], int, int],
    projection_id: str,
    projection_kind: str,
    authored_by: str,
    verified: bool,
    projection_uri: str,
    projection_sha256: str,
    projection_end: int,
) -> HflowProjectionRow:
    episode_id, mcap_uri, mcap_sha256, channels, start_ns, end_ns = episode
    quality_state = "quarantined" if report.quarantined else "accepted"
    return HflowProjectionRow.model_validate(
        {
            "bridge_schema_version": 1,
            "hflow_release": HFLOW_RELEASE,
            "hflow_revision": HFLOW_REVISION,
            "hflow_schema_version": report.stamps.schema_version,
            "pipeline_version": report.stamps.pipeline_version,
            "curation_query_digest": value_digest({"sql": CURATION_SQL}),
            "check_versions": (
                ProducerVersion(producer=f"hflow-check/{CHECK_NAME}", version=CHECK_VERSION),
            ),
            "enrichment_versions": (
                ProducerVersion(
                    producer=f"hflow-enrichment/{ENRICHMENT_NAME}",
                    version=ENRICHMENT_VERSION,
                ),
            ),
            "episode_id": episode_id,
            "mcap_uri": mcap_uri,
            "mcap_sha256": mcap_sha256,
            "channels": channels,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "quality_state": quality_state,
            "quarantine_tags": tuple(report.quarantine_tags),
            "projection_id": projection_id,
            "projection_kind": projection_kind,
            "authored_by": authored_by,
            "verified": verified,
            "verification_ref": "verification" if authored_by == "model" and verified else None,
            "language": "en",
            "projection_uri": projection_uri,
            "projection_sha256": projection_sha256,
            "projection_start": 0,
            "projection_end": projection_end,
        }
    )


def _projection_row(
    root: Path,
    report: Any,
    episode: tuple[str, str, str, tuple[str, ...], int, int],
    spec: tuple[str, str, str, bool, str],
) -> HflowProjectionRow:
    projection_id, projection_kind, authored_by, verified, text = spec
    projection_uri, projection_sha256 = _write_projection(root, projection_id, text)
    row = _row(
        report=report,
        episode=episode,
        projection_id=projection_id,
        projection_kind=projection_kind,
        authored_by=authored_by,
        verified=verified,
        projection_uri=projection_uri,
        projection_sha256=projection_sha256,
        projection_end=len(text),
    )
    if projection_id.startswith("unverified"):
        return row.model_copy(update={"quality_state": "unverified"})
    return row


def _write_verification(root: Path, row: HflowProjectionRow) -> None:
    source = root / row.projection_uri
    corpus_copy = root / "corpus" / row.projection_uri
    corpus_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, corpus_copy)
    text = source.read_text(encoding="utf-8")
    item = GoldItem(
        id=row.projection_id,
        lang=row.language,
        question="What procedure does this episode show?",
        reference_answer=text,
        source_doc_id=row.projection_uri,
        source_spans=[
            SourceSpan(
                doc_id=row.projection_uri,
                char_start=row.projection_start,
                char_end=row.projection_end,
                text=text,
            )
        ],
        provenance="human-verified",
        verified=True,
        split="calibration",
    )
    dump_goldset([item], root / "verification" / "goldset.jsonl")


def write_fixture_manifest(root: Path) -> HflowFixtureManifest:
    files = tuple(
        HflowFixtureFile(path=path.relative_to(root).as_posix(), sha256=file_digest(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != FIXTURE_MANIFEST_NAME
    )
    manifest = HflowFixtureManifest(
        schema_version=1,
        hflow_release=HFLOW_RELEASE,
        hflow_revision=HFLOW_REVISION,
        manifest=PROJECTION_MANIFEST_NAME,
        files=files,
    )
    atomic_write_text(
        root / FIXTURE_MANIFEST_NAME,
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    return manifest


def build_hflow_fixture(root: Path, clean_report: Any, bad_report: Any) -> HflowFixtureManifest:
    """Materialize MCAP, Parquet, projection, and verification boundary inputs."""
    if root.exists():
        raise ValueError(f"refusing to replace an existing HFlow fixture: {root}")
    root.mkdir(parents=True)
    clean_episode = _copy_episode(root, clean_report)
    bad_episode = _copy_episode(root, bad_report)
    rows = tuple(
        _projection_row(
            root,
            clean_report if not spec[0].startswith("quarantined") else bad_report,
            clean_episode if not spec[0].startswith("quarantined") else bad_episode,
            spec,
        )
        for spec in _PROJECTIONS
    )
    verified_model = next(row for row in rows if row.authored_by == "model" and row.verified)
    _write_verification(root, verified_model)
    write_projection_manifest(root / PROJECTION_MANIFEST_NAME, rows)
    return write_fixture_manifest(root)
