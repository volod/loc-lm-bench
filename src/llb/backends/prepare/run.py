"""The `prepare_models` orchestrator: detect the GPU, plan actions, run the disk preflight, and
execute (or, with dry_run, just plan) each row through the injectable fetchers.

The pure `plan`/`decide` logic lives in `planning`; the side-effecting `ollama_pull`/`hf_cache`
default to `fetch` but are injectable, so the whole flow is unit-testable with fakes.
"""

from pathlib import Path

from llb.backends import hardware
from llb.backends.hardware import Gpu, detect_gpus, max_vram_mb
from llb.backends.prepare.base import (
    ACTION_CACHE,
    ACTION_PULL,
    ACTION_SKIP,
    DiskFreeReader,
    HfCache,
    OllamaPull,
    PrepareProgress,
    PresentCheck,
)
from llb.backends.prepare.fetch import _hf_cache, _ollama_pull
from llb.backends.prepare.planning import acceptance_url, plan
from llb.backends.prepare.stores import (
    _default_present_check,
    disk_precheck,
    estimate_download_mb,
    store_dir_for,
)
from llb.core.contracts import ModelSpec, PreparationReport, PreparedModel


def _disk_status(
    row: PreparedModel,
    *,
    dry_run: bool,
    cache_dir: Path | None,
    free_reader: DiskFreeReader,
    present_check: PresentCheck,
) -> tuple[str | None, str]:
    """Disk preflight for one row: (block_reason, note).

    `block_reason` is a non-None string only when a real (non-dry) download must be refused for
    lack of space; the row is then failed before the long pull starts. `note` is an informational
    annotation ("cached (reuse)" or a dry-run preview of the shortfall) and never blocks.
    """
    if row["action"] not in (ACTION_PULL, ACTION_CACHE):
        return None, ""
    if present_check(row):
        return None, "cached (reuse)"
    store = store_dir_for(row["backend"], cache_dir)
    ok, reason = disk_precheck(estimate_download_mb(row), free_reader(store))
    if ok:
        return None, ""
    return (None, reason) if dry_run else (reason, "")


def _prepare_row_status(
    row: PreparedModel,
    *,
    dry_run: bool,
    ollama_pull: OllamaPull,
    hf_cache: HfCache,
    token: str | None,
    cache_dir: Path | None,
) -> tuple[str, str]:
    """Run or plan one manifest row; return (status, detail)."""
    if dry_run or row["action"] == ACTION_SKIP:
        status = (
            "planned" if dry_run else ("skipped" if row["action"] == ACTION_SKIP else "planned")
        )
        return status, row["reason"]
    if row["action"] == ACTION_PULL:
        ok, detail = ollama_pull(row["source"])
        return ("done" if ok else "failed"), detail
    ok, detail = hf_cache(row["source"], token, cache_dir)
    return ("done" if ok else "failed"), detail


def _annotate_detail(detail: str, row: ModelSpec, disk_note: str | None) -> str:
    """Append the disk headroom note and the license-acceptance URL to a row's detail."""
    if disk_note:
        detail = f"{detail}  [disk: {disk_note}]" if detail else f"[disk: {disk_note}]"
    url = acceptance_url(row)
    if url and "huggingface.co" not in detail:
        detail = f"{detail}  [license: {url}]"
    return detail


def prepare_models(
    models: list[ModelSpec],
    *,
    backend_filter: str = "all",
    force: bool = False,
    dry_run: bool = False,
    token: str | None = None,
    cache_dir: Path | None = None,
    gpus: list[Gpu] | None = None,
    ollama_pull: OllamaPull | None = None,
    hf_cache: HfCache | None = None,
    progress: PrepareProgress | None = None,
    disk_free_reader: DiskFreeReader | None = None,
    present_check: PresentCheck | None = None,
) -> PreparationReport:
    """Execute (or, with dry_run, just plan) model preparation. Returns a report dict."""
    gpus = detect_gpus() if gpus is None else gpus
    max_mb = max_vram_mb(gpus)
    rows = plan(models, max_mb, bool(gpus), backend_filter, force)
    ollama_pull = ollama_pull or _ollama_pull
    hf_cache = hf_cache or _hf_cache
    disk_free_reader = disk_free_reader or hardware.disk_free_mb
    present_check = present_check or _default_present_check

    results: list[PreparedModel] = []
    for row in rows:
        if progress is not None:
            progress(row)
        block, disk_note = _disk_status(
            row,
            dry_run=dry_run,
            cache_dir=cache_dir,
            free_reader=disk_free_reader,
            present_check=present_check,
        )
        if block is not None:
            results.append({**row, "status": "failed", "detail": block})
            continue
        status, detail = _prepare_row_status(
            row,
            dry_run=dry_run,
            ollama_pull=ollama_pull,
            hf_cache=hf_cache,
            token=token,
            cache_dir=cache_dir,
        )
        detail = _annotate_detail(detail, row, disk_note)
        results.append({**row, "status": status, "detail": detail})
    return {"gpus": gpus, "max_vram_mb": max_mb, "results": results}
