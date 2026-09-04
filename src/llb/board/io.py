"""Shared readers for persisted board run bundles."""

import json
import logging
from pathlib import Path
from typing import Any

from llb.artifacts.errors import ArtifactContractError
from llb.artifacts.run_bundle.manifests import read_run_manifest
from llb.core.contracts.common import JsonObject
from llb.core.contracts.run_bundle.manifest import RunManifestDocument

_LOG = logging.getLogger(__name__)


def admitted_manifest(manifest_path: Path) -> JsonObject | None:
    """One run head, admitted to the board only if it resolves through its own contract.

    A board ranks runs against each other, so what it must refuse is not a corrupt file -- that
    fails anyway -- but a bundle whose numbers mean something OTHER than the ones beside them: a
    manifest naming a family this build does not know, or a version it cannot read, or rows whose
    stamped identity contradicts what the manifest declared they were. Each is dropped with its
    reason rather than ranked, because a row silently admitted under the wrong reading is the one
    failure a leaderboard cannot show.

    A pre-contract bundle is not such a case: it is read at the version the family declares its
    history to be, migrated forward, and admitted like any other.
    """
    try:
        manifest = read_run_manifest(manifest_path)
        _refuse_mixed_score_rows(manifest_path.parent, manifest)
    except (OSError, ArtifactContractError) as exc:
        _LOG.warning("[board] refusing run bundle %s: %s", manifest_path.parent, exc)
        return None
    return manifest.model_dump(mode="json")


def _refuse_mixed_score_rows(run_dir: Path, manifest: RunManifestDocument) -> None:
    """Refuse a bundle whose rows stamp an identity its manifest did not declare.

    Only the first stamped row is read. A score file is written in one pass by one producer, so a
    single row is enough to tell a bundle whose members agree from one whose members were mixed,
    and the board pays one line rather than one validation per case.
    """
    contract = manifest.score_rows.record_contract if manifest.score_rows is not None else None
    scores = run_dir / "scores.jsonl"
    if contract is None or not scores.is_file():
        return
    for line in scores.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stamped = isinstance(row, dict) and row.get("schema_id") is not None
        if stamped and (row["schema_id"], row.get("schema_version")) != (
            contract.schema_id,
            contract.schema_version,
        ):
            raise ArtifactContractError(
                f"{scores}: rows are {row['schema_id']}@{row.get('schema_version')}, "
                f"manifest declares {contract.schema_id}@{contract.schema_version}"
            )
        return


def read_case_rows(path: Path) -> list[dict[str, Any]]:
    """Load a run bundle's canonical per-case rows from its `scores.jsonl`.

    `run_eval()['rows']` holds the aggregate leaderboard row, not the per-case ones, so any lane
    that compares two runs item by item reads them back from this file. A row without `item_id`
    is a different artifact shape and raises rather than silently comparing nothing.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or "item_id" not in value:
                raise ValueError(f"{path}:{line_number}: expected a per-case score row")
            rows.append(value)
    return rows


def read_case_series(run_dir: Path, column: str) -> list[float]:
    """Per-case values of one score column from the run bundle's `scores.jsonl`."""
    jsonl = run_dir / "scores.jsonl"
    out: list[float] = []
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get(column)
            if value is not None:
                out.append(float(value))
    return out


def read_case_objectives(run_dir: Path) -> list[float]:
    """Per-case objective scores for bootstrap CIs."""
    return read_case_series(run_dir, "objective_score")


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
