"""Start one book-ocr lane and read how long its recognition calls took.

The campaign decides the order. This module is the process and the GPU wait.
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.data_prep.book import BookPageManifest, PageLedgerRecord, PageStageState
from llb.prep.book.config import RunRequest
from llb.prep.book.constants import BLOCKS_FILENAME, CONFIG_FILENAME, PAGES_DIRNAME
from llb.prep.book.ranges import page_id
from llb.prep.book.runner import resume_run, start_run
from llb.prep.book.stages.recognize import RECOGNITION_STAGES

_LEDGER_STAGES = frozenset((*RECOGNITION_STAGES, "blocks"))


@dataclass(frozen=True)
class LaneRun:
    """What one finished lane directory holds."""

    run_dir: Path
    seconds_per_page: float | None
    failed_pages: list[int]
    errors: list[str]


@dataclass(frozen=True)
class CampaignHooks:
    """The process boundary tests replace. The default starts real engines."""

    start: Callable[[RunRequest, Path, Path, str], LaneRun]
    used_mb: Callable[[], int | None]
    reclaim: Callable[[int | None], None]


def default_hooks() -> CampaignHooks:
    """Start book-ocr runs and wait for the GPU to clear between them."""
    return CampaignHooks(start_lane, _used_mb, _reclaim)


def lane_timing(
    run_dir: Path, manifest: BookPageManifest
) -> tuple[float | None, list[int], list[str]]:
    """Mean seconds of one recognition call, and the pages that did not finish."""
    failed: list[int] = []
    errors: list[str] = []
    samples: list[float] = []
    for page in manifest.pages:
        _collect(run_dir, manifest.source_page_count, page, failed, errors, samples)
    seconds = round(sum(samples) / len(samples), 3) if samples else None
    return seconds, failed, errors


def _collect(
    run_dir: Path,
    page_count: int,
    page: PageLedgerRecord,
    failed: list[int],
    errors: list[str],
    samples: list[float],
) -> None:
    stages = [stage for stage in page.stages if stage.stage in _LEDGER_STAGES]
    problem = _problem(run_dir, page_count, page.pdf_page, stages)
    if problem is not None:
        failed.append(page.pdf_page)
        errors.extend(problem)
        return
    calls = [stage for stage in stages if stage.stage in RECOGNITION_STAGES]
    if calls:
        samples.append(sum(stage.duration_s for stage in calls) / len(calls))


def _problem(
    run_dir: Path, page_count: int, pdf_page: int, stages: list[PageStageState]
) -> list[str] | None:
    bad = [stage for stage in stages if stage.status != "done"]
    if bad:
        return [stage.error or stage.status for stage in bad]
    if _ledger(run_dir, pdf_page, page_count).is_file():
        return None
    return [f"pdf page {pdf_page} has no block ledger"]


def start_lane(request: RunRequest, out_root: Path, run_dir: Path, run_id: str) -> LaneRun:
    """Resume a lane that already has a config, otherwise start it in ``run_dir``."""
    del out_root
    if (run_dir / CONFIG_FILENAME).is_file():
        outcome = resume_run(run_dir)
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        outcome = start_run(request, run_dir.parent, run_dir=run_dir, run_id=run_id)
    seconds, failed, errors = lane_timing(outcome.run_dir, outcome.manifest)
    return LaneRun(outcome.run_dir, seconds, failed, errors)


def _ledger(run_dir: Path, pdf_page: int, page_count: int) -> Path:
    return run_dir / PAGES_DIRNAME / page_id(pdf_page, page_count) / BLOCKS_FILENAME


def _used_mb() -> int | None:
    from llb.prep.book.survey.measure import device_used_reader

    reader = device_used_reader()
    if reader is None:
        return None
    return reader()


def _reclaim(baseline_mb: int | None) -> None:
    from llb.prep.book.survey.measure import wait_for_reclaim

    wait_for_reclaim(baseline_mb)
