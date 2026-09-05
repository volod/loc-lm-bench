"""Read and write the head of a run bundle through its registered contract."""

import json
from pathlib import Path
from collections.abc import Mapping, Sequence

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import DatasetReadError
from llb.artifacts.io import json_document
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.artifacts import ContractReference
from llb.core.contracts.run_bundle.manifest import (
    RUN_MANIFEST_SCHEMA_ID,
    RunManifestDocument,
    ScoreRowsDeclaration,
)

MANIFEST_FILE = "manifest.json"
SCORES_FILE = "scores.jsonl"
RETRIEVAL_FILE = "retrieval.jsonl"


def read_run_manifest(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> RunManifestDocument:
    """Read one `manifest.json`, current or pre-contract, at the current contract version."""
    path = Path(path)
    read = registry.read_as(RUN_MANIFEST_SCHEMA_ID, json_document(path), source=str(path))
    assert isinstance(read, RunManifestDocument)
    return read


def declare_score_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    schema_id: str | None = None,
    owner: str | None = None,
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> ScoreRowsDeclaration:
    """Declare what the rows about to be published answer to, and check that they do.

    A method whose row shape this project models names its registered contract, and every row is
    validated against it here -- before publication, where a refusal costs a run that has not been
    published rather than a board reading that cannot be trusted. A benchmark STUDY names itself
    and the exact column set it wrote, which is the strongest claim available for a table whose
    columns that study chose.
    """
    if (schema_id is None) == (owner is None):
        raise ValueError("score rows declare either a registered contract or an owning study")
    if schema_id is not None:
        definition = registry.definition(schema_id)
        for index, row in enumerate(rows, start=1):
            registry.read_as(
                schema_id,
                dict(row),
                version=definition.current_version if "schema_id" in row else None,
                source=f"{SCORES_FILE}#record-{index}",
            )
        return ScoreRowsDeclaration(
            record_contract=ContractReference(
                schema_id=schema_id, schema_version=definition.current_version
            )
        )
    columns = sorted({str(key) for row in rows for key in row})
    return ScoreRowsDeclaration(owner=owner, columns=columns)


def read_score_rows(
    path: Path | str,
    declaration: ScoreRowsDeclaration | None,
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> int:
    """Read every score row against what the bundle declared it wrote, returning the count.

    A bundle that declares a contract has each row dispatched through it. A bundle that declares
    an owning study has each row checked against the column set that study published: a row
    carrying a column the run never declared is a different table under the same name, which is
    exactly the confusion the declaration exists to prevent.
    """
    path = Path(path)
    rows = _rows(path)
    if declaration is None:
        return len(rows)
    contract = declaration.record_contract
    if contract is not None:
        for index, row in enumerate(rows, start=1):
            registry.read_as(
                contract.schema_id,
                row,
                version=contract.schema_version if "schema_id" in row else None,
                source=f"{path}#record-{index}",
            )
        return len(rows)
    declared = set(declaration.columns or ())
    for index, row in enumerate(rows, start=1):
        undeclared = sorted(set(row) - declared)
        if undeclared:
            raise DatasetReadError(
                f"{path}#record-{index}: columns {undeclared} were not declared by "
                f"{declaration.owner!r}"
            )
    return len(rows)


def _rows(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetReadError(f"{path}: cannot read score rows: {exc}") from exc
    rows: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetReadError(f"{path}:{number}: unreadable row: {exc}") from exc
        if not isinstance(value, dict):
            raise DatasetReadError(f"{path}:{number}: expected one object record")
        rows.append(value)
    return tuple(rows)
