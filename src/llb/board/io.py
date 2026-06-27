"""Shared readers for persisted board run bundles."""

import json
from pathlib import Path


def read_case_splits(run_dir: Path) -> set[str]:
    """Read represented splits for legacy manifests that predate the manifest `split` field."""
    jsonl = run_dir / "scores.jsonl"
    if jsonl.exists():
        splits: set[str] = set()
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get("split")
            if isinstance(value, str):
                splits.add(value)
        return splits
    parquet = run_dir / "scores.parquet"
    if parquet.exists():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet, columns=["split"])
            return {str(value) for value in table.column("split").to_pylist() if value is not None}
        except Exception:  # pragma: no cover - optional dep / legacy schema drift
            return set()
    return set()


def read_case_series(run_dir: Path, column: str) -> list[float]:
    """Per-case values of one score column, preferring JSONL with Parquet fallback."""
    jsonl = run_dir / "scores.jsonl"
    if jsonl.exists():
        out: list[float] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get(column)
            if value is not None:
                out.append(float(value))
        return out
    parquet = run_dir / "scores.parquet"
    if parquet.exists():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet)
            if column not in table.column_names:
                return []
            return [float(v) for v in table.column(column).to_pylist() if v is not None]
        except Exception:  # pragma: no cover - optional dep / schema drift
            return []
    return []


def read_case_objectives(run_dir: Path) -> list[float]:
    """Per-case objective scores for bootstrap CIs."""
    return read_case_series(run_dir, "objective_score")


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
