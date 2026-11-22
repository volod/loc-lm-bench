"""Start, resume, and drive a `book-ocr` run.

A run walks its pages in chapter-sized batches; inside a batch it runs each configured stage over
every page before the next stage, so a model-backed stage loads its model once per batch rather
than once per page. After every batch -- and on the way out of an interrupted run -- the page
manifest and the processing report are rewritten from the stage records on disk.
"""

import logging
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from llb.core.contracts.data_prep.book import (
    BookOcrRunConfig,
    BookPageManifest,
    BookStageSpec,
    PageInventoryRecord,
    SourcePolicy,
)
from llb.prep.book.cache import ReuseIndex, discard_debris
from llb.prep.book.config import (
    Preflight,
    RunRequest,
    stage_errors,
    build_config,
    preflight,
    read_config,
    write_config,
)
from llb.prep.book.constants import RUN_TIMESTAMP_FORMAT
from llb.prep.book.inventory import inventory_page, open_pdf
from llb.prep.book.ledger import build_manifest, write_manifest
from llb.prep.book.logs import log_event, run_log
from llb.prep.book.models import OllamaLister
from llb.prep.book.ranges import batches, page_id
from llb.prep.book.report import write_report
from llb.prep.book.source import (
    place_source,
    refuse_source_overlap,
    stored_source_path,
    verify_source,
)
from llb.prep.book.stage_exec import PageWork, run_stage
from llb.prep.book.stages import StageRegistry, default_registry


class BookRunError(ValueError):
    """The run cannot start; every reason is in `errors`."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class RunOutcome:
    run_dir: Path
    manifest: BookPageManifest
    outcomes: dict[str, int] = field(default_factory=dict)  # ran / cached / reused / failed / ...


def new_run_dir(out_root: Path) -> tuple[str, Path]:
    """A fresh `(run_id, <out_root>/<run_timestamp>)`, never an existing directory."""
    run_id = uuid.uuid4().hex[:12]
    stamp = datetime.now(timezone.utc).strftime(RUN_TIMESTAMP_FORMAT)
    return run_id, out_root / f"{stamp}-{run_id}"


def check_request(
    request: RunRequest,
    registry: StageRegistry | None = None,
    ollama_lister: OllamaLister | None = None,
) -> Preflight:
    """The dry run: every refusal a real start would raise, with nothing written."""
    return preflight(request, registry or default_registry(), ollama_lister)


def start_run(
    request: RunRequest,
    out_root: Path,
    registry: StageRegistry | None = None,
    ollama_lister: OllamaLister | None = None,
) -> RunOutcome:
    """Validate, create the run directory, hold the source, record the config, and run."""
    registry = registry or default_registry()
    plan = preflight(request, registry, ollama_lister)
    if not plan.ok or plan.sha256 is None or plan.page_count is None:
        raise BookRunError(plan.errors)
    run_id, run_dir = new_run_dir(out_root)
    refuse_source_overlap(request.source, run_dir)
    run_dir.mkdir(parents=True)
    policy = cast(SourcePolicy, request.source_policy)  # preflight refused any other value
    source = place_source(request.source, run_dir, policy, plan.sha256, plan.page_count)
    config = build_config(request, plan, source, run_id)
    write_config(run_dir, config)
    return execute(run_dir, config, registry)


def resume_run(
    run_dir: Path,
    given_source: Path | None = None,
    registry: StageRegistry | None = None,
    ollama_lister: OllamaLister | None = None,
) -> RunOutcome:
    """Continue `run_dir` under its recorded config, refusing a source that no longer matches."""
    registry = registry or default_registry()
    config = read_config(run_dir)
    verify_source(run_dir, config.source, given_source)
    errors = stage_errors(config.stages, registry, ollama_lister)
    if errors:
        raise BookRunError(errors)
    return execute(run_dir, config, registry)


def execute(run_dir: Path, config: BookOcrRunConfig, registry: StageRegistry) -> RunOutcome:
    """Bring every configured stage on every selected page to a committed state."""
    outcomes: Counter[str] = Counter()
    with run_log(run_dir):
        removed = discard_debris(run_dir)
        index = ReuseIndex(Path(config.reuse_from) if config.reuse_from else None)
        log_event(
            "run_start",
            logging.INFO,
            run_id=config.run_id,
            pages=len(config.pages),
            stages=[spec.name for spec in config.stages],
            discarded_partials=removed,
            reusable_stages=len(index),
        )
        document = open_pdf(stored_source_path(run_dir, config.source))
        try:
            inventories = {page: inventory_page(document, page) for page in config.pages}
            manifest = build_manifest(run_dir, config, inventories, registry)
            write_manifest(run_dir, manifest)
            try:
                _run_batches(run_dir, config, inventories, registry, outcomes, index, document)
            finally:
                manifest = _publish(run_dir, config, inventories, registry)
        finally:
            document.close()
        log_event("run_end", logging.INFO, run_id=config.run_id, **dict(sorted(outcomes.items())))
    return RunOutcome(run_dir=run_dir, manifest=manifest, outcomes=dict(outcomes))


def _run_batches(
    run_dir: Path,
    config: BookOcrRunConfig,
    inventories: Mapping[int, PageInventoryRecord],
    registry: StageRegistry,
    outcomes: Counter[str],
    index: ReuseIndex,
    document: Any,
) -> None:
    for number, batch in enumerate(batches(config.pages, config.batch_pages)):
        log_event("batch_start", logging.INFO, batch=number, first=batch[0], last=batch[-1])
        for spec in config.stages:
            _drive_stage(
                run_dir, config, inventories, registry, spec, batch, outcomes, index, document
            )
        _publish(run_dir, config, inventories, registry)


def _drive_stage(
    run_dir: Path,
    config: BookOcrRunConfig,
    inventories: Mapping[int, PageInventoryRecord],
    registry: StageRegistry,
    spec: BookStageSpec,
    batch: list[int],
    outcomes: Counter[str],
    index: ReuseIndex,
    document: Any,
) -> None:
    """One stage across one batch. The stage may hold a server open for these pages only."""
    stage = registry.get(spec.name)
    names = [item.name for item in config.stages]
    before = tuple(names[: names.index(spec.name)])
    opener = getattr(stage, "open_batch", None)
    closer = getattr(stage, "close_batch", None)
    if opener is not None:
        opener()
    try:
        order = list(batch)
        order_fn = getattr(stage, "batch_order", None)
        if order_fn is not None:
            order = order_fn(run_dir, config, batch, spec)
        for pdf_page in order:
            work = PageWork(
                run_dir=run_dir,
                page_id=page_id(pdf_page, config.source.page_count),
                pdf_page=pdf_page,
                inventory=inventories[pdf_page],
                source_sha256=config.source.sha256,
                document=document,
            )
            outcomes[run_stage(work, stage, spec, config.max_attempts, index, before)] += 1
    finally:
        if closer is not None:
            closer()


def _publish(
    run_dir: Path,
    config: BookOcrRunConfig,
    inventories: Mapping[int, PageInventoryRecord],
    registry: StageRegistry,
) -> BookPageManifest:
    manifest = build_manifest(run_dir, config, inventories, registry)
    write_manifest(run_dir, manifest)
    write_report(run_dir, config, manifest)
    return manifest
