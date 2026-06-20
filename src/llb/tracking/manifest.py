"""Canonical run record (manifest + per-case scores), MLflow as a mirror only.

Correctness contract (design): the immutable manifest (JSON) and the per-case scores are
written to `$DATA_DIR` FIRST; only then is MLflow mirrored, best-effort. So a store/MLflow
error can never lose a completed run, and the canonical record never depends on MLflow
being installed. Scores go to Parquet when `pyarrow` (the `[track]` extra) is present, and
fall back to JSONL otherwise, so the base install still records everything.

`persist_run` takes an injectable `mirror` callable, so "manifest-before-mirror" ordering
and "mirror failure does not lose data" are both unit-testable without MLflow.
"""

import json
import logging
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

_LOG = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_env() -> dict:
    """Minimal reproducibility environment (GPU/driver added with telemetry in M2)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


class RunManifest(BaseModel):
    """Immutable per-run record: config + environment + headline metrics."""

    run_id: str
    run_name: str
    created_at: str = Field(default_factory=_utc_now)
    config: dict
    env: dict = Field(default_factory=capture_env)
    metrics: dict = Field(default_factory=dict)
    retrieval: dict = Field(default_factory=dict)
    judge: dict = Field(default_factory=dict)
    telemetry: dict = Field(default_factory=dict)
    n_cases: int = 0


def write_scores(rows: list[dict], path_no_ext: Path) -> Path:
    """Write per-case scores. Parquet if pyarrow is available, else JSONL. Returns path."""
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        out = path_no_ext.with_suffix(".jsonl")
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        _atomic_write_text(out, content)
        return out
    out = path_no_ext.with_suffix(".parquet")
    with tempfile.NamedTemporaryFile(
        dir=out.parent, prefix=f".{out.name}.", suffix=".tmp", delete=False
    ) as temp:
        temp_path = Path(temp.name)
    try:
        pq.write_table(pa.Table.from_pylist(rows), str(temp_path))
        temp_path.replace(out)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return out


def persist_run(
    manifest: RunManifest,
    case_rows: list[dict],
    out_dir: Path | str,
    mirror: Callable[[RunManifest, Path], None] | None = None,
) -> dict:
    """Write manifest + scores FIRST, then mirror (best-effort). Returns written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    score_paths = (out_dir / "scores.jsonl", out_dir / "scores.parquet")
    if manifest_path.exists() or any(path.exists() for path in score_paths):
        raise FileExistsError(f"run artifacts already exist in {out_dir}")
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2),
    )
    scores_path = write_scores(case_rows, out_dir / "scores")

    # Mirror only after the canonical record exists on disk; never let it raise.
    mirror = mirror if mirror is not None else mlflow_mirror
    mirror_status = "skipped"
    try:
        mirror(manifest, out_dir)
        mirror_status = "ok"
    except Exception as exc:  # a mirror failure must not lose a completed run
        mirror_status = f"failed: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"

    return {
        "manifest": str(manifest_path),
        "scores": str(scores_path),
        "mirror": mirror_status,
    }


def mlflow_mirror(manifest: RunManifest, out_dir: Path) -> None:
    """Mirror a manifest into a local MLflow file store. Needs the `[track]` extra."""
    try:
        import mlflow
    except ImportError:
        _LOG.info("[tracking] mlflow not installed; skipping mirror (canonical record on disk).")
        return
    # MLflow 3.x deprecated the local file store and raises unless opted in. The design
    # explicitly allows local file/SQLite mode (no server), so we opt in.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((out_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment("loc-lm-bench")
    with mlflow.start_run(run_name=manifest.run_name):
        mlflow.log_params({k: v for k, v in manifest.config.items() if _scalar(v)})
        mlflow.log_metrics({k: float(v) for k, v in manifest.metrics.items() if _numeric(v)})
        mlflow.log_artifact(str(out_dir / "manifest.json"))


def _scalar(value) -> bool:
    return isinstance(value, (str, int, float, bool))


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace ``path`` with UTF-8 text using a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temp.write(content)
            temp_path = Path(temp.name)
        temp_path.replace(path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
