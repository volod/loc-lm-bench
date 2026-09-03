"""Open a published run bundle: its manifest, its score rows, its retrieval rows.

Every lane that compares two runs, ranks a board, exports a fine-tuning set, or analyses misses
reads the same three files, and each used to do it with its own `json.loads` loop. One reader per
member is what makes a refusal mean something: a bundle from a build this one cannot read is named
HERE, at the door, instead of being aggregated with the half of it this reader understands.

Identity is stripped on the way in, so every caller sees exactly the flat record its producer
wrote -- `manifest["metrics"]`, `row["objective_score"]`, `row["item_id"]` -- whether the bundle
was published before these contracts existed or after.
"""

import json
from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import DatasetReadError
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.runs.datasets import (
    KIND_BENCHMARK,
    KIND_RUN,
    MANIFEST_FILE,
    RETRIEVAL_FILE,
    SCORES_FILE,
)
from llb.artifacts.runs.rows import decode_record, read_rows
from llb.core.contracts.common import JsonObject
from llb.core.contracts.run_bundle import STUDY_ANALYSIS_SCHEMA_ID, STUDY_DESIGN_SCHEMA_ID
from llb.core.contracts.runs import RUN_MANIFEST_SCHEMA_ID


def read_run_manifest(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> JsonObject:
    """One run bundle's `manifest.json` as its readers use it, refusing what this build cannot read.

    `path` is the manifest itself or the bundle directory holding it.
    """
    target = Path(path)
    target = target / MANIFEST_FILE if target.is_dir() else target
    record = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise DatasetReadError(f"{target}: a run manifest must be one object")
    return decode_record(RUN_MANIFEST_SCHEMA_ID, record, source=str(target), registry=registry)


# What a benchmark category run records about itself and an evaluation run does not: which
# category it is a cell of. `persist_category_run` writes it into every bundle's config.
CATEGORY_KEY = "category"


def run_bundle_kind(run_dir: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY) -> str:
    """Which score contract a bundle's rows are bound to, read off the bundle itself.

    A bundle this build published says so in its own description; one written before descriptions
    existed says it in its manifest, because a benchmark category run records the category it is a
    cell of and an evaluation run has none.

    A manifest this build cannot read answers `run` -- deciding which contract to describe the
    rows at is not the place to refuse a bundle, and reporting every member's refusal is more use
    to an operator than one error about the manifest.
    """
    try:
        config = read_run_manifest(run_dir, registry).get("config") or {}
    except (OSError, ValueError):
        return KIND_RUN
    return KIND_BENCHMARK if CATEGORY_KEY in config else KIND_RUN


def read_score_rows(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> list[JsonObject]:
    """A bundle's per-case rows, flat, whichever score contract the bundle was published under.

    An evaluation bundle's rows are `llb.case-score` and a benchmark bundle's are
    `llb.benchmark-cell`; both read back as the flat columns their lane wrote, so a caller that
    only wants `objective_score` never has to know which kind of bundle it opened.
    """
    target = Path(path)
    target = target / SCORES_FILE if target.is_dir() else target
    return read_rows(target, registry=registry)


def read_retrieval_rows(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> list[JsonObject]:
    """A bundle's per-case retrieved-span rows, flat."""
    target = Path(path)
    target = target / RETRIEVAL_FILE if target.is_dir() else target
    return read_rows(target, registry=registry)


def read_case_rows(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> list[JsonObject]:
    """A bundle's canonical PER-CASE rows, for a lane that compares two runs item by item.

    `run_eval()["rows"]` holds the aggregate leaderboard row, not the per-case ones, so any lane
    that pairs two runs reads them back from the file. A row without `item_id` is a different
    artifact shape and raises rather than silently comparing nothing.
    """
    target = Path(path)
    rows = read_score_rows(target, registry)
    for number, row in enumerate(rows, start=1):
        if "item_id" not in row:
            raise ValueError(f"{target}:{number}: expected a per-case score row")
    return rows


def read_study_design(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> JsonObject:
    """One `llb.study-design` sidecar's declared design, current form or pre-contract."""
    return _envelope_body(Path(path), STUDY_DESIGN_SCHEMA_ID, registry)


def read_study_analysis(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> JsonObject:
    """One `llb.study-analysis` sidecar's reading, current form or pre-contract.

    A sidecar written before the envelope existed IS the bare analysis, with nowhere to put an
    identity; the family declares which field it became, so both forms read the same body.
    """
    return _envelope_body(Path(path), STUDY_ANALYSIS_SCHEMA_ID, registry)


def _envelope_body(path: Path, schema_id: str, registry: ContractRegistry) -> JsonObject:
    return decode_record(
        schema_id, json.loads(path.read_text(encoding="utf-8")), source=str(path), registry=registry
    )


def read_case_series(
    run_dir: Path | str, column: str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> list[float]:
    """Per-case values of one numeric score column, skipping the cases that recorded none."""
    scores = Path(run_dir) / SCORES_FILE
    if not scores.is_file():
        return []
    values = (row.get(column) for row in read_score_rows(scores, registry))
    return [float(value) for value in values if value is not None]
