"""Run one configured stage on one page: skip it, reuse it, or execute it with bounded retries.

The decision is made from the cache key alone. A committed record under the same key whose
outputs still hash to what it recorded is complete, so it is skipped; an earlier run's record under
the same key is copied in; anything else runs. An `Exception` from a stage is a failure of that
page's stage -- retried up to the configured attempts, then recorded -- and never of the job. A
`BaseException` (Ctrl-C, a kill the interpreter sees) is not caught: the attempt's `.partial`
directory is all it leaves, and the resume deletes it.
"""

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llb.core.contracts.data_prep.book import BookStageRecord, BookStageSpec, PageInventoryRecord
from llb.prep.book.cache import (
    ReuseIndex,
    commit,
    digest_json,
    fresh_partial,
    list_outputs,
    outputs_digest,
    outputs_intact,
    partial_dir,
    read_record,
    stage_dir,
    stage_key,
)
from llb.prep.book.constants import (
    CONTRACT_VERSION,
    STAGE_RECORD_FILENAME,
    STAGE_RECORD_SCHEMA_ID,
    STATUS_DONE,
    STATUS_FAILED,
)
from llb.prep.book.logs import log_event
from llb.prep.book.stages import Stage, StageContext

if TYPE_CHECKING:
    import fitz

OUTCOME_CACHED = "cached"
OUTCOME_REUSED = "reused"
OUTCOME_RAN = "ran"
OUTCOME_FAILED = "failed"
OUTCOME_BLOCKED = "blocked"


@dataclass(frozen=True)
class PageWork:
    """One page as every stage on it sees it."""

    run_dir: Path
    page_id: str
    pdf_page: int
    inventory: PageInventoryRecord
    source_sha256: str
    document: "fitz.Document"


@dataclass(frozen=True)
class _Keyed:
    key: str
    tool: str
    params: dict[str, Any]
    params_digest: str
    input_digests: dict[str, str]
    inputs: dict[str, Path]


def _keyed(work: PageWork, stage: Stage, spec: BookStageSpec) -> _Keyed | None:
    """The stage's cache key on this page, or None when an upstream stage is not complete."""
    digests: dict[str, str] = {}
    inputs: dict[str, Path] = {}
    if stage.reads_source:
        digests["source"] = work.source_sha256
        digests["pdf_page"] = str(work.pdf_page)
    for upstream in stage.inputs:
        directory = stage_dir(work.run_dir, work.page_id, upstream)
        record = read_record(directory)
        if record is None or record.status != STATUS_DONE:
            return None
        digests[upstream] = outputs_digest(record)
        inputs[upstream] = directory
    tool = stage.tool_identity()
    params = stage.effective_params(spec.params, work.inventory, work.document, work.pdf_page)
    params_digest = digest_json(params)
    key = stage_key(
        stage=stage.name,
        stage_version=stage.version,
        tool=tool,
        model=spec.model,
        params_digest=params_digest,
        input_digests=digests,
    )
    return _Keyed(key, tool, params, params_digest, digests, inputs)


def _record(
    work: PageWork, stage: Stage, spec: BookStageSpec, keyed: _Keyed, **outcome: object
) -> BookStageRecord:
    return BookStageRecord.model_validate(
        {
            "schema_id": STAGE_RECORD_SCHEMA_ID,
            "schema_version": CONTRACT_VERSION,
            "kind": "book-ocr-stage-record",
            "page_id": work.page_id,
            "pdf_page": work.pdf_page,
            "stage": stage.name,
            "cache_key": keyed.key,
            "stage_version": stage.version,
            "tool": keyed.tool,
            "model": spec.model,
            "params_digest": keyed.params_digest,
            "input_digests": keyed.input_digests,
            **outcome,
        }
    )


def _reuse(work: PageWork, stage: Stage, keyed: _Keyed, index: ReuseIndex) -> bool:
    hit = index.lookup(keyed.key)
    if hit is None:
        return False
    source_dir, earlier = hit
    partial = partial_dir(work.run_dir, work.page_id, stage.name)
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(source_dir, partial, ignore=shutil.ignore_patterns(STAGE_RECORD_FILENAME))
    record = earlier.model_copy(
        update={
            "page_id": work.page_id,
            "pdf_page": work.pdf_page,
            "reused_from": str(index.run_dir),
        }
    )
    commit(partial, stage_dir(work.run_dir, work.page_id, stage.name), record)
    return True


def run_stage(
    work: PageWork, stage: Stage, spec: BookStageSpec, max_attempts: int, index: ReuseIndex
) -> str:
    """Bring one stage on one page to a committed state; returns what happened."""
    fields: dict[str, Any] = {
        "page_id": work.page_id,
        "pdf_page": work.pdf_page,
        "stage": stage.name,
    }
    keyed = _keyed(work, stage, spec)
    if keyed is None:
        log_event("stage_blocked", **fields)
        return OUTCOME_BLOCKED
    final = stage_dir(work.run_dir, work.page_id, stage.name)
    existing = read_record(final)
    if (
        existing is not None
        and existing.status == STATUS_DONE
        and existing.cache_key == keyed.key
        and outputs_intact(final, existing)
    ):
        log_event("stage_cached", cache_key=keyed.key, **fields)
        return OUTCOME_CACHED
    if _reuse(work, stage, keyed, index):
        log_event("stage_reused", cache_key=keyed.key, reused_from=str(index.run_dir), **fields)
        return OUTCOME_REUSED
    error = ""
    for attempt in range(1, max_attempts + 1):
        partial = fresh_partial(work.run_dir, work.page_id, stage.name)
        started = time.monotonic()
        log_event("stage_start", attempt=attempt, cache_key=keyed.key, **fields)
        ctx = StageContext(
            page_id=work.page_id,
            pdf_page=work.pdf_page,
            inventory=work.inventory,
            params=keyed.params,
            model=spec.model,
            inputs=keyed.inputs,
            work_dir=partial,
            document=work.document,
        )
        try:
            result = stage.run(ctx)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log_event("stage_attempt_failed", attempt=attempt, error=error, **fields)
            continue
        duration = round(time.monotonic() - started, 3)
        record = _record(
            work,
            stage,
            spec,
            keyed,
            status=STATUS_DONE,
            attempts=attempt,
            duration_s=duration,
            outputs=[out.model_dump() for out in list_outputs(partial)],
            meta=dict(result.meta),
        )
        commit(partial, final, record)
        log_event("stage_done", attempt=attempt, duration_s=duration, **fields)
        return OUTCOME_RAN
    partial = fresh_partial(work.run_dir, work.page_id, stage.name)
    record = _record(
        work,
        stage,
        spec,
        keyed,
        status=STATUS_FAILED,
        attempts=max_attempts,
        duration_s=0.0,
        error=error,
    )
    commit(partial, final, record)
    log_event("stage_failed", logging.WARNING, attempts=max_attempts, error=error, **fields)
    return OUTCOME_FAILED
