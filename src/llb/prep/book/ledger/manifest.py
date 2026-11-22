"""The page manifest: the committed stage records projected into one per-page, per-stage ledger.

The stage records on disk are the truth; the manifest is rebuilt from them, so `status` after a
kill reports exactly what a resume will find, even when the last manifest write predates the kill.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from llb.core.contracts.data_prep.book import (
    BookOcrRunConfig,
    BookPageManifest,
    PageInventoryRecord,
    PageLedgerRecord,
    PageStageState,
)
from llb.core.fsutil import atomic_write_text
from llb.prep.book.cache import read_record, stage_dir
from llb.prep.book.constants import (
    CONTRACT_VERSION,
    METADATA_DIRNAME,
    PAGE_MANIFEST_FILENAME,
    PAGE_MANIFEST_SCHEMA_ID,
    PAGE_STATUSES,
    STATUS_BLOCKED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
)
from llb.prep.book.ranges import batches, page_id
from llb.prep.book.stages import StageRegistry, resolved_inputs
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE

REUSED = "reused"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / METADATA_DIRNAME / PAGE_MANIFEST_FILENAME


def read_manifest(run_dir: Path) -> BookPageManifest | None:
    path = manifest_path(run_dir)
    if not path.is_file():
        return None
    try:
        return BookPageManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None


def _upstream(
    registry: StageRegistry | None,
    name: str,
    configured: Sequence[str],
    params_of: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    """Stages whose failure blocks `name`. A stage may choose them from what is configured."""
    before = tuple(configured[: configured.index(name)])
    if registry is not None and name in registry:
        return resolved_inputs(registry.get(name), before, params_of.get(name))
    return before  # unknown stage: assume it reads everything ahead of it


def page_states(
    run_dir: Path,
    pid: str,
    stage_names: Sequence[str],
    registry: StageRegistry | None,
    params_of: Mapping[str, Mapping[str, object]],
) -> tuple[list[PageStageState], str | None]:
    """Every configured stage's state on one page, and the page-image digest when rendered."""
    states: dict[str, PageStageState] = {}
    image_sha: str | None = None
    for name in stage_names:
        record = read_record(stage_dir(run_dir, pid, name))
        if record is None:
            upstream_failed = any(
                states[up].status in (STATUS_FAILED, STATUS_BLOCKED)
                for up in _upstream(registry, name, stage_names, params_of)
                if up in states
            )
            states[name] = PageStageState(
                stage=name,
                status=STATUS_BLOCKED if upstream_failed else STATUS_PENDING,
                attempts=0,
                duration_s=0.0,
            )
            continue
        states[name] = PageStageState(
            stage=name,
            status=record.status,
            cache_key=record.cache_key,
            attempts=record.attempts,
            duration_s=record.duration_s,
            error=record.error,
            reused_from=record.reused_from,
        )
        if name == RENDER_STAGE and record.status == STATUS_DONE:
            image_sha = next(
                (out.sha256 for out in record.outputs if out.path == PAGE_IMAGE_FILENAME), None
            )
    return [states[name] for name in stage_names], image_sha


def _summary(
    pages: Sequence[PageLedgerRecord], stage_names: Sequence[str]
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for name in stage_names:
        counts: Counter[str] = Counter({status: 0 for status in PAGE_STATUSES})
        for page in pages:
            state = next(s for s in page.stages if s.stage == name)
            counts[state.status] += 1
            counts[REUSED] += int(state.reused_from is not None)
        summary[name] = dict(sorted(counts.items()))
    return summary


def build_manifest(
    run_dir: Path,
    config: BookOcrRunConfig,
    inventories: Mapping[int, PageInventoryRecord],
    registry: StageRegistry | None,
) -> BookPageManifest:
    """Project the stage records of every configured page into the page manifest."""
    stage_names = [spec.name for spec in config.stages]
    params_of = {spec.name: spec.params for spec in config.stages}
    count = config.source.page_count
    pages = []
    for batch_index, batch in enumerate(batches(config.pages, config.batch_pages)):
        for pdf_page in batch:
            pid = page_id(pdf_page, count)
            states, image_sha = page_states(run_dir, pid, stage_names, registry, params_of)
            pages.append(
                PageLedgerRecord(
                    page_id=pid,
                    pdf_page=pdf_page,
                    batch=batch_index,
                    inventory=inventories[pdf_page],
                    image_sha256=image_sha,
                    stages=states,
                )
            )
    return BookPageManifest(
        schema_id=PAGE_MANIFEST_SCHEMA_ID,
        schema_version=CONTRACT_VERSION,
        kind="book-ocr-page-manifest",
        run_id=config.run_id,
        source_sha256=config.source.sha256,
        source_page_count=count,
        n_pages=len(pages),
        stage_names=stage_names,
        summary=_summary(pages, stage_names),
        pages=pages,
    )


def write_manifest(run_dir: Path, manifest: BookPageManifest) -> None:
    atomic_write_text(
        manifest_path(run_dir),
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )


def refresh_manifest(
    run_dir: Path, registry: StageRegistry | None, config: BookOcrRunConfig
) -> BookPageManifest:
    """Rebuild the manifest from disk, reusing the inventory the last manifest recorded."""
    previous = read_manifest(run_dir)
    if previous is None:
        raise ValueError(
            f"{run_dir} has no readable page manifest; rerun with --resume to rebuild it"
        )
    inventories = {page.pdf_page: page.inventory for page in previous.pages}
    return build_manifest(run_dir, config, inventories, registry)
