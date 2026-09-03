"""Publish a run bundle atomically, with MLflow as a mirror only.

Correctness contract (design): the immutable manifest (JSON) and the per-case scores are
written to `$DATA_DIR` FIRST; only then is MLflow mirrored, best-effort. So a store/MLflow
error can never lose a completed run, and the canonical record never depends on MLflow
being installed. Scores are always JSONL -- a single, zero-dep format so a run bundle is
identical across environments and never branches on which optional extras are installed.

Every member is written through its registered contract and the whole staged directory is
described and read back BEFORE the rename, so a bundle that reaches `$DATA_DIR` is one this build
could read again. Additional members are `RunMember`s rather than a `name -> text` map: each says
which contract validates it, or that it is the declared human-report exemption.

`persist_run` takes an injectable `mirror` callable, so "manifest-before-mirror" ordering
and "mirror failure does not lose data" are both unit-testable without MLflow.
"""

import json
import logging
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from llb.artifacts.datasets import publish_dataset_manifest
from llb.artifacts.errors import DatasetReadError
from llb.artifacts.runs.datasets import CASE_CONTRACTS, KIND_RUN, run_bundle_manifest
from llb.artifacts.runs.members import RunMember, member_problems
from llb.artifacts.runs.rows import write_rows
from llb.core.contracts.run_bundle import CASE_RETRIEVAL_SCHEMA_ID, CASE_SCORE_SCHEMA_ID
from llb.core.contracts.runs import RunManifest, RunPaths
from llb.core.fsutil import atomic_write_text as _atomic_write_text

_LOG = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
SCORES_STEM = "scores"
RETRIEVAL_STEM = "retrieval"


def write_scores(
    rows: Sequence[Mapping[str, object]],
    path_no_ext: Path,
    schema_id: str = CASE_SCORE_SCHEMA_ID,
) -> Path:
    """Write per-case rows as JSONL through their contract (deterministic, zero-dep).

    JSONL is the single canonical on-disk format so a run bundle is identical regardless of which
    optional extras happen to be installed -- the artifact never branches on `[track]`/pyarrow.
    A row that does not satisfy `schema_id` fails here, before anything is published.
    """
    return write_rows(Path(path_no_ext).with_suffix(".jsonl"), schema_id, rows)


def persist_run(
    manifest: RunManifest,
    case_rows: Sequence[Mapping[str, object]],
    out_dir: Path | str,
    mirror: Callable[[RunManifest, Path], None] | None = None,
    staging_dir: Path | str | None = None,
    retrieval_rows: Sequence[Mapping[str, object]] | None = None,
    artifacts: Sequence[RunMember] | None = None,
    kind: str = KIND_RUN,
) -> RunPaths:
    """Atomically publish manifest, scores, and declared members as one described directory.

    `retrieval_rows` is the additive `retrieval.jsonl` record used by miss analysis. `artifacts`
    adds declared members to the same staging transaction. `kind` says which contract the score
    rows satisfy -- an evaluation's case scores or a benchmark lane's cells. The staged directory
    is described, read back member by member, and given its `dataset_manifest.json` before the
    rename, so a half-readable bundle never becomes visible. The external mirror remains
    best-effort and starts only after the complete canonical bundle is visible.
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        raise FileExistsError(f"run artifacts already exist in {out_dir}")
    members = tuple(artifacts or ())
    problems = member_problems(members)
    if problems:
        raise ValueError("invalid additional run bundle member(s):\n- " + "\n- ".join(problems))

    staging = _staging_dir(out_dir, staging_dir)
    try:
        staged_scores = _stage_bundle(staging, manifest, case_rows, retrieval_rows, members, kind)
        staging.replace(out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    paths: RunPaths = {
        "manifest": str(out_dir / MANIFEST_FILE),
        "scores": str(out_dir / staged_scores.name),
        "mirror": _mirrored(manifest, out_dir, mirror),
    }
    if retrieval_rows is not None:
        paths["retrieval"] = str(out_dir / f"{RETRIEVAL_STEM}.jsonl")
    return paths


def _staging_dir(out_dir: Path, staging_dir: Path | str | None) -> Path:
    staging = (
        Path(staging_dir)
        if staging_dir is not None
        else Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{out_dir.name}.tmp-"))
    )
    if staging.parent.resolve() != out_dir.parent.resolve():
        raise ValueError("staging_dir must be a sibling of out_dir for atomic publication")
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def _stage_bundle(
    staging: Path,
    manifest: RunManifest,
    case_rows: Sequence[Mapping[str, object]],
    retrieval_rows: Sequence[Mapping[str, object]] | None,
    members: Sequence[RunMember],
    kind: str,
) -> Path:
    """Write every member into `staging`, then refuse the whole bundle unless all read back."""
    staging_manifest = staging / MANIFEST_FILE
    if staging_manifest.exists() or any(staging.glob("scores.*")):
        raise FileExistsError(f"staged canonical artifacts already exist in {staging}")
    _atomic_write_text(
        staging_manifest,
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
    )
    staged_scores = write_scores(case_rows, staging / SCORES_STEM, _case_contract(kind))
    if retrieval_rows is not None:
        write_rows(staging / f"{RETRIEVAL_STEM}.jsonl", CASE_RETRIEVAL_SCHEMA_ID, retrieval_rows)
    for member in members:
        target = staging / member.name
        if target.exists():
            raise ValueError(f"additional member would overwrite a staged file: {member.name!r}")
        _atomic_write_text(target, member.content)
    _publish_description(staging, kind, tuple(members))
    return staged_scores


def _case_contract(kind: str) -> str:
    try:
        return CASE_CONTRACTS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown run bundle kind {kind!r}") from exc


def _publish_description(staging: Path, kind: str, members: tuple[RunMember, ...]) -> None:
    """Describe the staged bundle, read every member back, and publish the description.

    This is the gate the rename is behind: a member that cannot be read at the current contract,
    or whose bytes do not match what was just described, refuses here -- while the only thing that
    exists is a staging directory the caller is about to delete.
    """
    from llb.artifacts.dataset_reading import survey_dataset

    description = run_bundle_manifest(staging, kind=kind, extra=members)
    refusals = [
        f"{reading.path}: {reading.refusal}"
        for reading in survey_dataset(staging, description)
        if reading.refusal
    ]
    if refusals:
        raise DatasetReadError(
            "staged run bundle cannot be read back at the current contracts:\n- "
            + "\n- ".join(refusals)
        )
    publish_dataset_manifest(staging, description)


def _mirrored(
    manifest: RunManifest, out_dir: Path, mirror: Callable[[RunManifest, Path], None] | None
) -> str:
    """Mirror only after the canonical record exists on disk; never let it raise."""
    mirror = mirror if mirror is not None else mlflow_mirror
    try:
        mirror(manifest, out_dir)
    except Exception as exc:  # a mirror failure must not lose a completed run
        return f"failed: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    return "ok"


def mlflow_mirror(manifest: RunManifest, out_dir: Path) -> None:
    """Mirror a manifest into the shared local MLflow SQLite store."""
    try:
        from llb.tracking.mlflow import mirror_run

        mirror_run(manifest, out_dir)
    except ImportError:
        _LOG.info("[tracking] mlflow not installed; skipping mirror (canonical record on disk).")
