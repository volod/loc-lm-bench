"""The content-addressed stage cache: keys, committed stage directories, and atomic commits.

A stage's key is the SHA-256 of its name and version, its tool identity, its model, the digest of
its parameters, and the digests of everything it reads -- the source and PDF page for a stage that
reads the PDF, the upstream stages' output digests otherwise. The render key is a function of the
source bytes, and a recognition key is a function of the page-image digest. A stage whose output
names the page (the ledger, the signal file) also includes the PDF page, so identical images do
not share that file.

A stage writes into `pages/<pNNNN>/.<stage>.partial/`, its record goes in last, and the whole
directory is renamed to `pages/<pNNNN>/<stage>/`. A reader therefore sees a complete stage or no
stage: a kill leaves only a `.partial` directory, which nothing reads and a resume deletes.
"""

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llb.core.contracts.data_prep.book import BookStageRecord, StageOutputRecord
from llb.core.fsutil import atomic_write_text
from llb.prep.book.constants import (
    PAGES_DIRNAME,
    PARTIAL_SUFFIX,
    STAGE_RECORD_FILENAME,
    STATUS_DONE,
    TRASH_SUFFIX,
)


def digest_json(value: Any) -> str:
    """Canonical SHA-256 of a JSON-able value (sorted keys, no whitespace)."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stage_key(
    *,
    stage: str,
    stage_version: str,
    tool: str,
    model: str | None,
    params_digest: str,
    input_digests: Mapping[str, str],
) -> str:
    """The cache key: one digest over every component a stage's output depends on."""
    return digest_json(
        {
            "stage": stage,
            "stage_version": stage_version,
            "tool": tool,
            "model": model,
            "params_digest": params_digest,
            "inputs": dict(input_digests),
        }
    )


def outputs_digest(record: BookStageRecord) -> str:
    """One digest over a committed stage's output files -- what a downstream key reads."""
    return digest_json([[out.path, out.sha256] for out in record.outputs])


def page_dir(run_dir: Path, page_id: str) -> Path:
    return run_dir / PAGES_DIRNAME / page_id


def stage_dir(run_dir: Path, page_id: str, stage: str) -> Path:
    return page_dir(run_dir, page_id) / stage


def partial_dir(run_dir: Path, page_id: str, stage: str) -> Path:
    return page_dir(run_dir, page_id) / f".{stage}{PARTIAL_SUFFIX}"


def read_record(directory: Path) -> BookStageRecord | None:
    """The stage record in `directory`, or None when absent or unreadable."""
    path = directory / STAGE_RECORD_FILENAME
    if not path.is_file():
        return None
    try:
        return BookStageRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None


def list_outputs(work_dir: Path) -> list[StageOutputRecord]:
    """Every file a stage wrote, sorted by relative path, with its digest and size."""
    outputs = []
    for path in sorted(p for p in work_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(work_dir).as_posix()
        if rel == STAGE_RECORD_FILENAME:
            continue
        data = path.read_bytes()
        outputs.append(
            StageOutputRecord(
                path=rel, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
            )
        )
    return outputs


def outputs_intact(directory: Path, record: BookStageRecord) -> bool:
    """True when every recorded output is present and still hashes to its recorded digest."""
    return [(out.path, out.sha256) for out in list_outputs(directory)] == [
        (out.path, out.sha256) for out in record.outputs
    ]


def fresh_partial(run_dir: Path, page_id: str, stage: str) -> Path:
    """An empty work directory for one attempt, discarding any earlier interrupted one."""
    work = partial_dir(run_dir, page_id, stage)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    return work


def commit(work_dir: Path, final_dir: Path, record: BookStageRecord) -> None:
    """Write the record into `work_dir` last, then swap the directory into place atomically."""
    atomic_write_text(
        work_dir / STAGE_RECORD_FILENAME,
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )
    trash = final_dir.with_name(f".{final_dir.name}{TRASH_SUFFIX}")
    if trash.exists():
        shutil.rmtree(trash)
    if final_dir.exists():
        final_dir.rename(trash)
    work_dir.rename(final_dir)
    if trash.exists():
        shutil.rmtree(trash)


def discard_debris(run_dir: Path) -> int:
    """Remove every uncommitted attempt and half-swapped directory a kill left behind."""
    removed = 0
    pages = run_dir / PAGES_DIRNAME
    if not pages.is_dir():
        return 0
    for debris in pages.glob("*/.*"):
        if debris.is_dir() and debris.name.endswith((PARTIAL_SUFFIX, TRASH_SUFFIX)):
            shutil.rmtree(debris)
            removed += 1
    return removed


class ReuseIndex:
    """Completed stages of an earlier run, addressed by cache key."""

    def __init__(self, run_dir: Path | None) -> None:
        self.run_dir = run_dir
        self._by_key: dict[str, Path] = {}
        if run_dir is None:
            return
        for record_path in sorted((run_dir / PAGES_DIRNAME).glob(f"*/*/{STAGE_RECORD_FILENAME}")):
            record = read_record(record_path.parent)
            if record is not None and record.status == STATUS_DONE:
                self._by_key.setdefault(record.cache_key, record_path.parent)

    def __len__(self) -> int:
        return len(self._by_key)

    def lookup(self, key: str) -> tuple[Path, BookStageRecord] | None:
        """The earlier committed directory for `key`, when its outputs are still intact."""
        directory = self._by_key.get(key)
        if directory is None:
            return None
        record = read_record(directory)
        if record is None or record.cache_key != key or not outputs_intact(directory, record):
            return None
        return directory, record
