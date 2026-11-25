"""Run the pilot roster one engine at a time and record what each one did.

Scoring stays in ``compare``. This module only produces the ledgers and the
existing lane's sidecar. A lane that finishes no recognition call is not
measured. Failed pages on a lane that did read are named, so an unread page
cannot win once the pages are scored.
"""

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

from llb.prep.book.compare.baseline import extract_baseline
from llb.prep.book.compare.campaign_host import CampaignHooks, LaneRun, default_hooks
from llb.prep.book.compare.campaign_record import (
    CampaignBaseline,
    CampaignLane,
    CampaignRecord,
    LaneStatus,
    write_campaign,
)
from llb.prep.book.compare.constants import STATUS_OVER_VRAM, STATUS_UNAVAILABLE
from llb.prep.book.compare.lane_config import lane_request
from llb.prep.book.compare.roster import PILOT_ROSTER, RosterEntry, partition
from llb.prep.book.engines.protocol import EngineUnavailable
from llb.prep.book.runner import BookRunError, new_run_dir

_LOG = logging.getLogger(__name__)
_PENDING: Literal["pending"] = "pending"
_RUNNING: Literal["running"] = "running"
_MEASURED: Literal["measured"] = "measured"
_DETAIL_LIMIT = 500
_ROLE = ("page-reader", "variant-policy")
_Role = Literal["page-reader", "variant-policy"]
_LANE_STATUS = ("pending", "running", "measured", "over-vram", "unavailable")


def prepare_campaign(
    *,
    pdf: Path,
    pilot: Path,
    digest: str,
    pages: Sequence[int],
    language: str,
    reuse_from: Path | None,
    host_mb: int | None,
    class_counts: dict[str, int],
    entries: Sequence[RosterEntry] = PILOT_ROSTER,
    problem: Callable[[str], str | None] | None = None,
) -> CampaignRecord:
    """The lanes this host will start, and the ones it will not."""
    ask = problem or engine_problem
    _runnable, skipped = partition(entries, host_mb)
    refused = {item.entry.lane: item.detail for item in skipped}
    lanes = [_initial(entry, refused, ask) for entry in entries]
    return CampaignRecord(
        language=language,
        source=str(pdf),
        source_sha256=digest,
        pilot=str(pilot),
        reuse_from=None if reuse_from is None else str(reuse_from),
        host_vram_mb=host_mb,
        class_counts=dict(class_counts),
        pages=list(pages),
        baseline=CampaignBaseline(status=_PENDING),
        lanes=lanes,
    )


def execute_campaign(
    record: CampaignRecord,
    bundle: Path,
    out_root: Path,
    hooks: CampaignHooks | None = None,
) -> CampaignRecord:
    """Run the baseline, then each unfinished lane. The index is saved as it goes."""
    active = hooks or default_hooks()
    record = _baseline(record, bundle, active)
    for index, lane in enumerate(record.lanes):
        if lane.status not in {_PENDING, _RUNNING}:
            continue
        record = _one(record, index, bundle, out_root, active)
    return record


def engine_problem(engine: str) -> str | None:
    """Why ``engine`` cannot start, or None when its adapter is ready."""
    from llb.prep.book.engines.registry import default_adapter_registry

    try:
        default_adapter_registry().require(engine).available()
    except EngineUnavailable as exc:
        return exc.reason
    return None


def roster_entry(lane: CampaignLane) -> RosterEntry:
    """The roster row a campaign lane was built from."""
    return RosterEntry(lane.lane, lane.engine, lane.role, lane.vram_mb, lane.variant, lane.dewarp)


def campaign_lines(record: CampaignRecord) -> list[str]:
    """A short ASCII summary. Page text stays in the run directories."""
    host = "no CUDA device" if record.host_vram_mb is None else f"{record.host_vram_mb} MB"
    lines = [f"[book-ocr] pilot language {record.language} host {host}"]
    base = record.baseline
    lines.append(_row("baseline", base.status, base.seconds_per_page, base.detail))
    for lane in record.lanes:
        lines.append(_row(lane.lane, lane.status, lane.seconds_per_page, lane.detail))
    return lines


def _initial(
    entry: RosterEntry, refused: dict[str, str], problem: Callable[[str], str | None]
) -> CampaignLane:
    detail = refused.get(entry.lane)
    if detail is not None:
        return _lane(entry, STATUS_OVER_VRAM, detail)
    reason = problem(entry.engine)
    if reason:
        return _lane(entry, STATUS_UNAVAILABLE, reason)
    return _lane(entry, _PENDING, "")


def _lane(entry: RosterEntry, status: str, detail: str) -> CampaignLane:
    if entry.role not in _ROLE:
        raise ValueError(f"unknown roster role {entry.role}")
    if status not in _LANE_STATUS:
        raise ValueError(f"unknown lane status {status}")
    return CampaignLane(
        lane=entry.lane,
        engine=entry.engine,
        role=cast(_Role, entry.role),
        vram_mb=entry.declared_vram_mb,
        variant=entry.variant,
        dewarp=entry.dewarp,
        status=cast(LaneStatus, status),
        detail=detail,
    )


def _baseline(record: CampaignRecord, bundle: Path, hooks: CampaignHooks) -> CampaignRecord:
    if record.baseline.status != _PENDING:
        return record
    _LOG.info("pilot baseline starting language %s", record.language)
    before = hooks.used_mb()
    started = time.monotonic()
    try:
        citations = extract_baseline(
            Path(record.source), list(record.pages), record.language, bundle / "baseline"
        )
    except (EngineUnavailable, OSError, ValueError) as exc:
        _LOG.error("pilot baseline failed: %s", exc)
        baseline = record.baseline.model_copy(
            update={"status": STATUS_UNAVAILABLE, "detail": str(exc)}
        )
        record = record.model_copy(update={"baseline": baseline})
        write_campaign(bundle, record)
        hooks.reclaim(before)
        return record
    elapsed = round((time.monotonic() - started) / len(record.pages), 3)
    baseline = record.baseline.model_copy(
        update={"status": _MEASURED, "citations": str(citations), "seconds_per_page": elapsed}
    )
    record = record.model_copy(update={"baseline": baseline})
    write_campaign(bundle, record)
    hooks.reclaim(before)
    return record


def _one(
    record: CampaignRecord,
    index: int,
    bundle: Path,
    out_root: Path,
    hooks: CampaignHooks,
) -> CampaignRecord:
    lane = record.lanes[index]
    run_id, run_dir = _destination(lane, out_root)
    record = _update(record, index, status=_RUNNING, run=str(run_dir))
    write_campaign(bundle, record)
    _LOG.info("pilot lane %s starting", lane.lane)
    before = hooks.used_mb()
    reuse = None if record.reuse_from is None else Path(record.reuse_from)
    request = lane_request(
        Path(record.source), record.pages, roster_entry(lane), record.language, reuse
    )
    try:
        result = hooks.start(request, out_root, run_dir, run_id)
    except BookRunError as exc:
        detail = "; ".join(exc.errors)[:_DETAIL_LIMIT]
        record = _update(record, index, status=STATUS_UNAVAILABLE, detail=detail)
        write_campaign(bundle, record)
        hooks.reclaim(before)
        return record
    status = _MEASURED if result.seconds_per_page is not None else STATUS_UNAVAILABLE
    detail = _failure(result) or ("no recognition call finished" if status != _MEASURED else "")
    record = _update(
        record,
        index,
        status=status,
        run=str(result.run_dir),
        seconds_per_page=result.seconds_per_page,
        failed_pages=list(result.failed_pages),
        detail=detail,
    )
    write_campaign(bundle, record)
    hooks.reclaim(before)
    return record


def _destination(lane: CampaignLane, out_root: Path) -> tuple[str, Path]:
    if lane.run:
        path = Path(lane.run)
        return path.name.rsplit("-", 1)[-1], path
    return new_run_dir(out_root)


def _update(record: CampaignRecord, index: int, **fields: object) -> CampaignRecord:
    lanes = list(record.lanes)
    lanes[index] = lanes[index].model_copy(update=fields)
    return record.model_copy(update={"lanes": lanes})


def _failure(result: LaneRun) -> str:
    if not result.failed_pages:
        return ""
    pages = ", ".join(str(page) for page in result.failed_pages)
    text = f"failed pdf pages {pages}"
    if result.errors:
        text = f"{text}: {result.errors[0]}"
    return text[:_DETAIL_LIMIT]


def _row(name: str, status: str, seconds: float | None, detail: str) -> str:
    rate = "" if seconds is None else f" {seconds:.3f}s/call"
    tail = f" {detail}" if detail else ""
    return f"[book-ocr] {status} {name}{rate}{tail}"
