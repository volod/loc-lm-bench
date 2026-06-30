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


def _check_jsonl(run_dir: Path, column: str) -> list[float]:
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


def _check_parquet(run_dir: Path, column: str) -> list[float]:
    parquet = run_dir / "scores.parquet"
    out: list[float] = []
    if parquet.exists():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet)
            if column in table.column_names:
                out = [float(v) for v in table.column(column).to_pylist() if v is not None]
        except Exception:  # pragma: no cover - optional dep / schema drift
            pass
    return out


def read_case_series(run_dir: Path, column: str) -> list[float]:
    """Per-case values of one score column, preferring JSONL with Parquet fallback."""
    out = _check_jsonl(run_dir, column)
    if out:
        return out

    return _check_parquet(run_dir, column)


def read_case_objectives(run_dir: Path) -> list[float]:
    """Per-case objective scores for bootstrap CIs."""
    return read_case_series(run_dir, "objective_score")


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
