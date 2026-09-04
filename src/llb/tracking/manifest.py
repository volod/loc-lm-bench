"""Canonical run record (manifest + per-case scores), MLflow as a mirror only.

Correctness contract (design): the immutable manifest (JSON) and the per-case scores are
written to `$DATA_DIR` FIRST; only then is MLflow mirrored, best-effort. So a store/MLflow
error can never lose a completed run, and the canonical record never depends on MLflow
being installed. Scores are always JSONL -- a single, zero-dep format so a run bundle is
identical across environments and never branches on which optional extras are installed.

`persist_run` takes an injectable `mirror` callable, so "manifest-before-mirror" ordering
and "mirror failure does not lose data" are both unit-testable without MLflow.

The bundle is also a CONTRACT (`llb.run-manifest`). A caller says what its score rows answer to
and hands over already-declared additional artifacts; publication reads every staged member back
through those declarations before the rename, so a bundle that reaches `$DATA_DIR` is one a board
or an external consumer can validate without guessing from filenames.
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

from pydantic import Field

from llb.artifacts.run_bundle.run_artifacts import RunArtifact
from llb.core.contracts.run_bundle.manifest import RunManifestDocument
from llb.core.contracts.runs import RunEnvironment, RunPaths
from llb.core.fsutil import atomic_write_text as _atomic_write_text

_LOG = logging.getLogger(__name__)

# The three files a bundle always publishes itself, which an additional artifact may not shadow.
CANONICAL_MEMBERS = frozenset({"manifest.json", "scores.jsonl", "retrieval.jsonl"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_env() -> RunEnvironment:
    """Minimal reproducibility environment (GPU/driver added with telemetry in backend telemetry)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


class RunManifest(RunManifestDocument):
    """The manifest as a producer builds it: the published contract, minus the busywork.

    It adds no field to `llb.run-manifest` and changes no meaning. The two defaults are the ones
    no producer should have to state -- when the run happened, and what interpreter it happened on
    -- so every bundle records them the same way rather than each caller remembering to.
    """

    created_at: str = Field(default_factory=_utc_now)
    env: RunEnvironment = Field(default_factory=capture_env)


def write_scores(rows: Sequence[Mapping[str, object]], path_no_ext: Path) -> Path:
    """Write per-case scores as JSONL (deterministic, zero-dep). Returns the path.

    JSONL is the single canonical on-disk format so a run bundle is identical regardless of which
    optional extras happen to be installed -- the artifact never branches on `[track]`/pyarrow.
    """
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    out = path_no_ext.with_suffix(".jsonl")
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(out, content)
    return out


def persist_run(
    manifest: RunManifest,
    case_rows: Sequence[Mapping[str, object]],
    out_dir: Path | str,
    mirror: Callable[[RunManifest, Path], None] | None = None,
    staging_dir: Path | str | None = None,
    retrieval_rows: Sequence[Mapping[str, object]] | None = None,
    artifacts: Sequence[RunArtifact] | None = None,
    *,
    score_contract: str | None = None,
    score_owner: str | None = None,
) -> RunPaths:
    """Atomically publish manifest, scores, and declared artifacts as one directory.

    `score_contract` names the registered row family the rows satisfy, or `score_owner` names the
    study whose own column set they are; exactly one is required, because a bundle that cannot say
    what its rows are is a bundle a later reader has to guess about. `retrieval_rows` is the
    additive `retrieval.jsonl` record used by miss analysis, and `artifacts` are the already-
    declared additional records. The external mirror remains best-effort and starts only after the
    complete canonical bundle is visible.
    """
    from llb.artifacts.run_bundle.manifests import declare_score_rows
    from llb.artifacts.run_bundle.publication import validate_staged_bundle

    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        raise FileExistsError(f"run artifacts already exist in {out_dir}")

    declared = list(artifacts or ())
    published = RunManifestDocument.model_validate(
        {
            **manifest.model_dump(),
            "score_rows": declare_score_rows(
                case_rows, schema_id=score_contract, owner=score_owner
            ).model_dump(),
            "artifacts": [artifact.declaration().model_dump() for artifact in declared],
        }
    )

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
            json.dumps(published.model_dump(mode="json"), ensure_ascii=False, indent=2),
        )
        staged_scores = write_scores(case_rows, staging / "scores")
        staged_retrieval = (
            write_scores(retrieval_rows, staging / "retrieval")
            if retrieval_rows is not None
            else None
        )
        for artifact in declared:
            # A canonical member is not something an additional artifact may replace: the
            # manifest, the score rows, and the retrieval sidecar are what the bundle IS.
            if artifact.name in CANONICAL_MEMBERS:
                raise ValueError(f"additional artifact may not be named {artifact.name!r}")
            _atomic_write_text(staging / artifact.name, artifact.content)
        validate_staged_bundle(staging)
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

    paths: RunPaths = {
        "manifest": str(manifest_path),
        "scores": str(scores_path),
        "mirror": mirror_status,
    }
    if staged_retrieval is not None:
        paths["retrieval"] = str(out_dir / staged_retrieval.name)
    return paths


def mlflow_mirror(manifest: RunManifest, out_dir: Path) -> None:
    """Mirror a manifest into the shared local MLflow SQLite store."""
    try:
        from llb.tracking.mlflow import mirror_run

        mirror_run(manifest, out_dir)
    except ImportError:
        _LOG.info("[tracking] mlflow not installed; skipping mirror (canonical record on disk).")
