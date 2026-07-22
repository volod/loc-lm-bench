"""Shared readers for persisted board run bundles."""

import json
from pathlib import Path
from typing import Any


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
