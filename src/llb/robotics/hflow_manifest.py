"""Read and write the portable Parquet manifest at the HFlow bridge boundary."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llb.robotics.digests import file_digest
from llb.robotics.evidence_models import (
    HflowFixtureManifest,
    HflowProjectionRow,
)
from llb.robotics.upstreams import HFLOW_RELEASE, HFLOW_REVISION

FIXTURE_MANIFEST_NAME = "fixture-manifest.json"
PROJECTION_MANIFEST_NAME = "manifest.parquet"

_COLUMNS = (
    "bridge_schema_version",
    "hflow_release",
    "hflow_revision",
    "hflow_schema_version",
    "pipeline_version",
    "curation_query_digest",
    "check_versions_json",
    "enrichment_versions_json",
    "episode_id",
    "mcap_uri",
    "mcap_sha256",
    "channels_json",
    "start_ns",
    "end_ns",
    "quality_state",
    "quarantine_tags_json",
    "projection_id",
    "projection_kind",
    "authored_by",
    "verified",
    "verification_ref",
    "language",
    "projection_uri",
    "projection_sha256",
    "projection_start",
    "projection_end",
)

_CREATE_SQL = """
CREATE TABLE projection_manifest (
    bridge_schema_version INTEGER NOT NULL,
    hflow_release VARCHAR NOT NULL,
    hflow_revision VARCHAR NOT NULL,
    hflow_schema_version VARCHAR NOT NULL,
    pipeline_version VARCHAR NOT NULL,
    curation_query_digest VARCHAR NOT NULL,
    check_versions_json VARCHAR NOT NULL,
    enrichment_versions_json VARCHAR NOT NULL,
    episode_id VARCHAR NOT NULL,
    mcap_uri VARCHAR NOT NULL,
    mcap_sha256 VARCHAR NOT NULL,
    channels_json VARCHAR NOT NULL,
    start_ns BIGINT NOT NULL,
    end_ns BIGINT NOT NULL,
    quality_state VARCHAR NOT NULL,
    quarantine_tags_json VARCHAR NOT NULL,
    projection_id VARCHAR NOT NULL,
    projection_kind VARCHAR NOT NULL,
    authored_by VARCHAR NOT NULL,
    verified BOOLEAN NOT NULL,
    verification_ref VARCHAR,
    language VARCHAR NOT NULL,
    projection_uri VARCHAR NOT NULL,
    projection_sha256 VARCHAR NOT NULL,
    projection_start BIGINT NOT NULL,
    projection_end BIGINT NOT NULL
)
"""


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "HFlow manifest support requires the robotics extra: uv pip install -e '.[robotics]'"
        ) from exc
    return duckdb


def resolve_fixture_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"fixture path must stay below its root: {relative}")
    return root / path


def load_fixture_manifest(root: Path) -> HflowFixtureManifest:
    root = Path(root).resolve()
    path = root / FIXTURE_MANIFEST_NAME
    try:
        fixture = HflowFixtureManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ValueError(f"{path}: invalid HFlow fixture manifest -- {exc}") from None
    if fixture.hflow_release != HFLOW_RELEASE or fixture.hflow_revision != HFLOW_REVISION:
        raise ValueError(
            f"HFlow fixture must pin {HFLOW_RELEASE}@{HFLOW_REVISION}, got "
            f"{fixture.hflow_release}@{fixture.hflow_revision}"
        )
    declared = [item.path for item in fixture.files]
    if len(declared) != len(set(declared)):
        raise ValueError("HFlow fixture manifest contains duplicate paths")
    observed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != FIXTURE_MANIFEST_NAME
    }
    if set(declared) != observed_files:
        raise ValueError("HFlow fixture files differ from its complete pinned file list")
    for item in fixture.files:
        observed = file_digest(resolve_fixture_path(root, item.path))
        if observed != item.sha256:
            raise ValueError(
                f"stale HFlow fixture file {item.path}: expected {item.sha256}, observed {observed}"
            )
    if fixture.manifest not in observed_files:
        raise ValueError("HFlow fixture does not pin its projection manifest")
    return fixture


def _decode_row(record: dict[str, Any]) -> HflowProjectionRow:
    payload = dict(record)
    for source, target in (
        ("check_versions_json", "check_versions"),
        ("enrichment_versions_json", "enrichment_versions"),
        ("channels_json", "channels"),
        ("quarantine_tags_json", "quarantine_tags"),
    ):
        raw = payload.pop(source)
        try:
            payload[target] = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"projection {payload.get('projection_id')}: invalid {source}"
            ) from exc
    try:
        return HflowProjectionRow.model_validate_json(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        )
    except ValidationError as exc:
        raise ValueError(
            f"projection {payload.get('projection_id')}: invalid HFlow manifest row -- {exc}"
        ) from None


def load_projection_manifest(path: Path) -> tuple[HflowProjectionRow, ...]:
    connection = _duckdb().connect()
    try:
        relation = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
        columns = tuple(item[0] for item in relation.description)
        if set(columns) != set(_COLUMNS) or len(columns) != len(_COLUMNS):
            raise ValueError(f"{path}: projection manifest columns differ from bridge schema v1")
        rows = [
            _decode_row(dict(zip(columns, values, strict=True))) for values in relation.fetchall()
        ]
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"{path}: projection manifest has no rows")
    projection_ids = [row.projection_id for row in rows]
    if len(projection_ids) != len(set(projection_ids)):
        raise ValueError(f"{path}: projection ids must be unique")
    return tuple(sorted(rows, key=lambda row: row.projection_id))


def _encoded_row(row: HflowProjectionRow) -> tuple[object, ...]:
    payload = row.model_dump(mode="json")
    return (
        payload["bridge_schema_version"],
        payload["hflow_release"],
        payload["hflow_revision"],
        payload["hflow_schema_version"],
        payload["pipeline_version"],
        payload["curation_query_digest"],
        json.dumps(payload["check_versions"], sort_keys=True, separators=(",", ":")),
        json.dumps(payload["enrichment_versions"], sort_keys=True, separators=(",", ":")),
        payload["episode_id"],
        payload["mcap_uri"],
        payload["mcap_sha256"],
        json.dumps(payload["channels"], separators=(",", ":")),
        payload["start_ns"],
        payload["end_ns"],
        payload["quality_state"],
        json.dumps(payload["quarantine_tags"], separators=(",", ":")),
        payload["projection_id"],
        payload["projection_kind"],
        payload["authored_by"],
        payload["verified"],
        payload["verification_ref"],
        payload["language"],
        payload["projection_uri"],
        payload["projection_sha256"],
        payload["projection_start"],
        payload["projection_end"],
    )


def write_projection_manifest(path: Path, rows: tuple[HflowProjectionRow, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = _duckdb().connect()
    try:
        connection.execute(_CREATE_SQL)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        connection.executemany(
            f"INSERT INTO projection_manifest VALUES ({placeholders})",
            [_encoded_row(row) for row in rows],
        )
        destination = str(path).replace("'", "''")
        connection.execute(
            f"COPY projection_manifest TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()
