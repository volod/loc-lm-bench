"""Format-aware readers for dataset members bound to registered row contracts."""

import csv
import hashlib
import json
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel

from llb.artifacts.errors import ArtifactContractError, DatasetReadError
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.artifacts import DatasetMember


def read_bound_member(
    dataset_root: Path,
    member: DatasetMember,
    registry: ContractRegistry,
) -> tuple[BaseModel, ...]:
    """Verify and read one manifest member through its declared record contract."""
    path = dataset_root / member.path
    if not path.is_file():
        if member.required:
            raise DatasetReadError(f"{path}: required dataset member is missing")
        return ()
    observed_digest = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    if observed_digest != member.digest:
        raise DatasetReadError(
            f"{path}: digest mismatch; manifest={member.digest}, observed={observed_digest}"
        )
    if member.format == "opaque":
        return ()
    expected = member.record_contract
    if expected is None:
        raise DatasetReadError(f"{path}: structured member has no record contract")
    records = _load_records(path, member.format, expected.schema_id, registry)
    validated: list[BaseModel] = []
    for index, record in enumerate(records, start=1):
        source = f"{path}#record-{index}"
        try:
            validated.append(
                registry.read_as(
                    expected.schema_id,
                    record,
                    version=expected.schema_version,
                    source=source,
                )
            )
        except ArtifactContractError as exc:
            raise DatasetReadError(
                f"{source}: manifest binds {expected.schema_id}@{expected.schema_version}; {exc}"
            ) from exc
    return tuple(validated)


def _load_records(
    path: Path, artifact_format: str, schema_id: str, registry: ContractRegistry
) -> tuple[dict[str, object], ...]:
    """Parse a member's records, normalizing whatever its family declares a pre-contract file to be.

    A whole-document member could be a bare array or map; a ROW member could be a bare body -- a
    benchmark cell's columns were written flat before the envelope around them existed. Both are
    the same declaration (`legacy_document_field`) and the same normalizer, so every parsed record
    passes through it and one that already carries an identity is returned untouched.
    """
    parsed = _parsed(path, artifact_format)
    return tuple(_normalized(record, path, schema_id, registry) for record in parsed)


def _parsed(path: Path, artifact_format: str) -> tuple[object, ...]:
    """A member's raw records, in whatever physical form it is bound to."""
    if artifact_format == "json":
        return (json.loads(path.read_text(encoding="utf-8")),)
    if artifact_format == "yaml":
        return (yaml.safe_load(path.read_text(encoding="utf-8")),)
    if artifact_format == "jsonl":
        return tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if artifact_format == "csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return tuple(cast(dict[str, object], row) for row in csv.DictReader(handle))
    if artifact_format == "parquet":
        return _load_parquet(path)
    raise DatasetReadError(f"{path}: unsupported structured format {artifact_format!r}")


def _load_parquet(path: Path) -> tuple[dict[str, object], ...]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DatasetReadError(
            f"{path}: Parquet reading requires the optional pyarrow package"
        ) from exc
    return tuple(cast(dict[str, object], row) for row in parquet.read_table(path).to_pylist())


def _normalized(
    value: object, path: Path, schema_id: str, registry: ContractRegistry
) -> dict[str, object]:
    try:
        return dict(registry.normalize(schema_id, value))
    except ArtifactContractError as exc:
        raise DatasetReadError(f"{path}: {exc}") from exc
