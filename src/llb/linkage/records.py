"""The record table a linkage run reads: JSONL in, a typed DuckDB table out.

Column types are DERIVED from the comparison kinds rather than inferred from the data, so the
same JSONL always materialises the same table -- an inferred `DATE` on one host and a `VARCHAR`
on another would silently change what the model was fitted on.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.core.contracts.common import JsonObject
from llb.linkage.constants import (
    DUCKDB_TYPES,
    KIND_COSINE,
    LABELS_TABLE,
    MATCH_LABELS_SUFFIX,
    RECORDS_TABLE,
)
from llb.linkage.spec import LinkageSpec


@dataclass(frozen=True)
class ReviewerLabel:
    """One reviewer decision about a record pair: same thing, or not."""

    left_id: str
    right_id: str
    match: bool = True

    @property
    def score(self) -> float:
        """Splink's `clerical_match_score`: this seam records binary reviewer decisions only."""
        return 1.0 if self.match else 0.0


def read_records(path: Path) -> list[JsonObject]:
    """Read a JSONL record table, skipping blank lines."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def column_types(spec: LinkageSpec) -> dict[str, str]:
    """DuckDB DDL types for the identifier, every compared column, and retained columns."""
    types: dict[str, str] = {spec.unique_id_column: "VARCHAR"}
    for comparison in spec.comparisons:
        types[comparison.column] = DUCKDB_TYPES[comparison.kind].format(
            dimension=comparison.dimension
        )
    for column in spec.retain_columns:
        types.setdefault(column, "VARCHAR")
    return types


def validate_records(records: Sequence[JsonObject], spec: LinkageSpec) -> None:
    """Fail early and by name on a table the spec cannot be applied to."""
    if not records:
        raise ValueError("the record table is empty")
    required = set(column_types(spec))
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"record {index} is missing column(s) {missing}")
        identifier = record.get(spec.unique_id_column)
        if identifier is None or str(identifier) == "":
            raise ValueError(f"record {index} has no {spec.unique_id_column!r} value")
        if str(identifier) in seen_ids:
            raise ValueError(f"duplicate {spec.unique_id_column} {identifier!r}")
        seen_ids.add(str(identifier))
        _validate_vector_widths(record, index, spec)


def _validate_vector_widths(record: JsonObject, index: int, spec: LinkageSpec) -> None:
    """A cosine column is a fixed-width array; a short row is a DDL error much further on."""
    for comparison in spec.comparisons:
        if comparison.kind != KIND_COSINE:
            continue
        vector = record.get(comparison.column)
        if vector is not None and len(vector) != comparison.dimension:
            raise ValueError(
                f"record {index} column {comparison.column!r} has {len(vector)} values, "
                f"but the comparison declares dimension {comparison.dimension}"
            )


def _cell(value: Any, sql_type: str) -> Any:
    """Coerce one JSON value to the DuckDB type its column was declared with."""
    if value is None:
        return None
    if sql_type == "VARCHAR[]":
        return [str(item) for item in value]
    if sql_type.startswith("DOUBLE["):
        return [float(item) for item in value]
    return str(value)


def register_records(
    con: Any, records: Sequence[JsonObject], spec: LinkageSpec, table: str = RECORDS_TABLE
) -> str:
    """Create and populate the run's record table on an open DuckDB connection."""
    validate_records(records, spec)
    types = column_types(spec)
    columns = ", ".join(f'"{name}" {sql_type}' for name, sql_type in types.items())
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    con.execute(f'CREATE TABLE "{table}" ({columns})')
    placeholders = ", ".join("?" for _ in types)
    rows = [
        tuple(_cell(record.get(name), sql_type) for name, sql_type in types.items())
        for record in records
    ]
    con.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', rows)
    return table


@dataclass(frozen=True)
class LabelTables:
    """The two views Splink needs of one reviewer label set.

    `judged` carries every decision with its `clerical_match_score` and is what the accuracy
    analysis reads. `matches` is the positives-only view, because
    `estimate_m_from_pairwise_labels` treats EVERY row it is given as a match -- handing it the
    non-matches would fit m on pairs a reviewer said were different records.
    """

    judged: str
    matches: str


def register_labels(
    con: Any, labels: Sequence["ReviewerLabel"], table: str = LABELS_TABLE
) -> LabelTables:
    """Register a reviewer label table: one row per judged pair, matches scored 1.0."""
    rows = sorted({(label.left_id, label.right_id, label.score) for label in labels})
    if not rows:
        raise ValueError("the reviewer label table holds no judged pairs")
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    con.execute(
        f'CREATE TABLE "{table}" '
        "(unique_id_l VARCHAR, unique_id_r VARCHAR, clerical_match_score DOUBLE)"
    )
    con.executemany(f'INSERT INTO "{table}" VALUES (?, ?, ?)', rows)
    matches = f"{table}{MATCH_LABELS_SUFFIX}"
    con.execute(f'DROP TABLE IF EXISTS "{matches}"')
    con.execute(
        f'CREATE TABLE "{matches}" AS SELECT unique_id_l, unique_id_r FROM "{table}" '
        "WHERE clerical_match_score >= 0.5"
    )
    if con.execute(f'SELECT count(*) FROM "{matches}"').fetchone()[0] == 0:
        raise ValueError("the reviewer label table holds no MATCHING pairs to fit m from")
    return LabelTables(judged=table, matches=matches)


def read_labels(path: Path, spec: LinkageSpec) -> list["ReviewerLabel"]:
    """Read reviewer decisions from JSONL rows of `{left, right, match}`.

    A `cannot-tell` decision is expressed by omitting the pair, not by a third value: the model
    can only be fitted or scored against pairs a reviewer actually settled.
    """
    identifier = spec.unique_id_column
    labels: list[ReviewerLabel] = []
    for row in read_records(Path(path)):
        left = row.get("left", row.get(f"{identifier}_l"))
        right = row.get("right", row.get(f"{identifier}_r"))
        if left is None or right is None:
            raise ValueError(f"label row {row!r} names no record pair")
        labels.append(
            ReviewerLabel(
                left_id=str(left), right_id=str(right), match=bool(row.get("match", True))
            )
        )
    if not labels:
        raise ValueError(f"no reviewer labels in {path}")
    return labels
