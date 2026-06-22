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
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from llb.contracts import (
    ContentionReport,
    JsonObject,
    JudgeStatus,
    RetrievalMetrics,
    RunEnvironment,
    RunMetrics,
    RunPaths,
    TelemetryReport,
)

_LOG = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_env() -> RunEnvironment:
    """Minimal reproducibility environment (GPU/driver added with telemetry in M2)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


class RunManifest(BaseModel):
    """Immutable per-run record: config + environment + headline metrics."""

    run_id: str
    run_name: str
    split: str | None = None
    created_at: str = Field(default_factory=_utc_now)
    config: JsonObject
    env: RunEnvironment = Field(default_factory=capture_env)
    metrics: RunMetrics | None = None
    retrieval: RetrievalMetrics | None = None
    judge: JudgeStatus | None = None
    telemetry: TelemetryReport | None = None
    contention: ContentionReport | None = None
    n_cases: int = 0


def write_scores(rows: Sequence[Mapping[str, object]], path_no_ext: Path) -> Path:
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
        pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), str(temp_path))
        temp_path.replace(out)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return out


def persist_run(
    manifest: RunManifest,
    case_rows: Sequence[Mapping[str, object]],
    out_dir: Path | str,
    mirror: Callable[[RunManifest, Path], None] | None = None,
    staging_dir: Path | str | None = None,
) -> RunPaths:
    """Atomically publish manifest + scores as one directory, then mirror best-effort."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        raise FileExistsError(f"run artifacts already exist in {out_dir}")

    staging = (
        Path(staging_dir)
        if staging_dir is not None
        else Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{out_dir.name}.tmp-"))
    )
    if staging.parent.resolve() != out_dir.parent.resolve():
        raise ValueError("staging_dir must be a sibling of out_dir for atomic publication")
    staging.mkdir(parents=True, exist_ok=True)

    try:
        staging_manifest = staging / "manifest.json"
        if staging_manifest.exists() or any(staging.glob("scores.*")):
            raise FileExistsError(f"staged canonical artifacts already exist in {staging}")
        _atomic_write_text(
            staging_manifest,
            json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2),
        )
        staged_scores = write_scores(case_rows, staging / "scores")
        staging.replace(out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest_path = out_dir / staging_manifest.name
    scores_path = out_dir / staged_scores.name

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
    """Mirror a manifest into the shared local MLflow SQLite store."""
    try:
        from llb.tracking.mlflow import mirror_run

        mirror_run(manifest, out_dir)
    except ImportError:
        _LOG.info("[tracking] mlflow not installed; skipping mirror (canonical record on disk).")


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
