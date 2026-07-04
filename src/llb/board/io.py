"""Shared readers for persisted board run bundles."""

import json
from pathlib import Path


def read_case_splits(run_dir: Path) -> set[str]:
    """Read represented splits for legacy manifests that predate the manifest `split` field."""
    jsonl = run_dir / "scores.jsonl"
    if not jsonl.exists():
        return set()
    splits: set[str] = set()
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line).get("split")
        if isinstance(value, str):
            splits.add(value)
    return splits


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
