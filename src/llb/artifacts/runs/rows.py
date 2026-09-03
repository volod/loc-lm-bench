"""Write and read the JSONL members of a run bundle through their registered contracts.

Identity lives on the FILE, not in the row -- the same seam `llb.artifacts.records` already draws
for chunk rows, and for the same reason: every consumer of a score row keys on `objective_score`,
`status`, and `item_id`, and adding two keys to each in-memory row would change what a few hundred
assertions, aggregations, and DataFrame projections see for no reading anyone takes.

A family whose body is the producer's own (`llb.benchmark-cell`) declares the field that body
lives under, which is the same field a pre-contract file's whole content became. Writing wraps the
row into it and reading unwraps it, so a benchmark row on disk names its contract while every lane
that reads one still sees the flat columns it wrote.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import ArtifactContractError, DatasetReadError
from llb.artifacts.records import decode, encode
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.common import JsonObject


def encode_record(
    schema_id: str, record: Mapping[str, Any], registry: ContractRegistry = DEFAULT_REGISTRY
) -> JsonObject:
    """One record as it is written: its identity, then the body its family declares.

    The encoded record is validated against the current contract before it is returned, so a
    producer that grew a column its contract does not declare fails at the write rather than
    leaving a bundle nobody can read back.
    """
    definition = registry.definition(schema_id)
    body_field = definition.legacy_document_field
    payload: Mapping[str, Any] = {body_field: dict(record)} if body_field else record
    encoded = encode(schema_id, definition.current_version, payload)
    registry.read_current(encoded, source=f"<{schema_id}>")
    return encoded


def decode_record(
    schema_id: str,
    record: object,
    *,
    source: str = "<record>",
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> JsonObject:
    """One record as its consumers see it: validated, without identity, body unwrapped.

    A stated `null` is dropped, in the body of an envelope family as well as in a modelled
    record's own columns, so one record reads the same however it was written -- see `_without_
    nulls`.
    """
    decoded = decode(schema_id, record, source=source, registry=registry)
    body_field = registry.definition(schema_id).legacy_document_field
    if body_field is None:
        return decoded
    body = decoded.get(body_field, {})
    return _without_nulls(body) if isinstance(body, Mapping) else {}


def encode_rows(
    schema_id: str,
    rows: Sequence[Mapping[str, Any]],
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> list[JsonObject]:
    """Every row of one JSONL member, encoded and validated before anything is written."""
    return [encode_record(schema_id, row, registry) for row in rows]


def read_rows(
    path: Path | str,
    schema_id: str | None = None,
    *,
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> list[JsonObject]:
    """Read one JSONL member of a run bundle, whatever contract state it is in.

    A row that names its family is read through it -- and a row naming a family or a version this
    build does not declare refuses here, at the reader every lane shares, rather than being scored
    with the half of it this build understands. A row carrying NO identity is a bundle this project
    wrote before the families existed: its flat columns are what every consumer already reads, so
    it is returned as written. `schema_id` names the family such a caller expects; without one, a
    row is read as whatever it declares itself to be.
    """
    target = Path(path)
    rows: list[JsonObject] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise DatasetReadError(f"{target}:{number}: expected one object record")
        rows.append(_row(raw, schema_id, f"{target}:{number}", registry))
    return rows


def _without_nulls(record: Mapping[str, Any]) -> JsonObject:
    """The record without the keys it states as absent.

    `null` here means "this producer recorded no such thing", which is what an absent key already
    means to every consumer -- `row.get("first_hit_rank")` answers None either way. Dropping it is
    what makes ONE reading of a row whether an old writer omitted the column, a current one wrote
    it out as null, or the column travels inside an envelope's open body.
    """
    return {key: value for key, value in record.items() if value is not None}


def _row(
    raw: JsonObject, schema_id: str | None, source: str, registry: ContractRegistry
) -> JsonObject:
    declared = raw.get("schema_id")
    if declared is None:
        return _without_nulls(raw)
    if not isinstance(declared, str):
        raise DatasetReadError(f"{source}: schema_id must be a string")
    if schema_id is not None and declared != schema_id:
        raise DatasetReadError(f"{source}: expected {schema_id!r}, observed {declared!r}")
    try:
        return decode_record(declared, raw, source=source, registry=registry)
    except ArtifactContractError as exc:
        raise DatasetReadError(str(exc)) from exc


def write_rows(
    path: Path | str,
    schema_id: str,
    rows: Sequence[Mapping[str, Any]],
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> Path:
    """Write one JSONL member, identity first, after every row has validated."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = encode_rows(schema_id, rows, registry)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in encoded), encoding="utf-8"
    )
    return target


def append_row(
    path: Path | str,
    schema_id: str,
    record: Mapping[str, Any],
    *,
    default: Any = None,
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> None:
    """Append one identified row to an append-only journal.

    `default` is the `json.dumps` fallback the caller already relies on for a value its own
    serializer produced (a numpy float retrieval score, a `Path`); the record still validates
    through the registry first.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = encode_record(schema_id, record, registry)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(encoded, ensure_ascii=False, default=default) + "\n")
