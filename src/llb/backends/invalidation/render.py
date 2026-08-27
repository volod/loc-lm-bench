"""Printing the invalidation report: what a swap voids, grouped by where an operator would fix it.

The findings are grouped by SURFACE rather than listed flat, because the three surfaces are three
different pieces of work -- a committed aggregate is re-run, a published value is restated out of
the new run, and a doc table row is edited -- and an operator sizing a swap is sizing those three
separately.

An empty finding list is printed as a sentence, never as an empty section. "Nothing is affected" is
the answer this report most often gives and the one most easily confused with a report that failed
to look, so the surfaces it walked and the record count each one held are printed either way.
"""

import json
from dataclasses import asdict

from llb.backends.invalidation.report import InvalidationReport
from llb.backends.invalidation.surfaces import (
    BASELINE_TABLES,
    COMMITTED_AGGREGATES,
    PUBLISHED_VALUES,
)

PREFIX = "[invalidation]"
_SURFACE_ORDER = (COMMITTED_AGGREGATES, PUBLISHED_VALUES, BASELINE_TABLES)
_CLEAN = "nothing published or committed was measured on the outgoing generation"
_ALL_PLACED = "every recorded model identity resolved to a registered family generation"


def _header(report: InvalidationReport) -> list[str]:
    return [
        f"{PREFIX} family {report.family_id} ({report.family_label}): {report.direction} of "
        f"generation `{report.target}` replaces carried generation `{report.outgoing}`"
    ]


def _surfaces(report: InvalidationReport) -> list[str]:
    lines = []
    for reading in report.readings:
        held = f"{len(reading.records)} record(s)"
        state = f"UNREAD -- {reading.error}" if reading.error else held
        lines.append(f"{PREFIX}   surface {reading.surface}: {reading.describe} -- {state}")
    return lines


def _findings(report: InvalidationReport) -> list[str]:
    if not report.invalidated:
        return [f"{PREFIX} {_CLEAN}."]
    lines = [
        f"{PREFIX} {len(report.invalidated)} record(s) across {len(report.entries)} roster "
        f"entry(ies) were measured on generation `{report.outgoing}` of {report.family_id}: "
        f"{', '.join(report.entries)}",
        f"{PREFIX} each record below is re-measured on `{report.target}` or restated from a run "
        "that was:",
    ]
    for surface in _SURFACE_ORDER:
        records = report.by_surface(surface)
        if not records:
            continue
        lines.append(f"{PREFIX}   {surface} ({len(records)})")
        lines.extend(f"{PREFIX}     {record.named()}" for record in records)
    return lines


def _unresolved(report: InvalidationReport) -> list[str]:
    if not report.unresolved:
        return [f"{PREFIX} {_ALL_PLACED}."]
    lines = [
        f"{PREFIX} {len(report.unresolved)} recorded identity(ies) the register cannot place -- "
        "read each by hand, because a model the roster dropped is invisible to this count:"
    ]
    lines.extend(f"{PREFIX}   {record.named()}" for record in report.unresolved)
    return lines


def render_text(report: InvalidationReport) -> str:
    """The operator-facing report: the swap, the surfaces walked, the findings, and the tally."""
    lines = _header(report) + _surfaces(report) + _findings(report) + _unresolved(report)
    lines.append(
        f"{PREFIX} {report.scanned} record(s) scanned across {len(report.readings)} surface(s), "
        f"{len(report.unread)} unread"
    )
    return "\n".join(lines)


def report_payload(report: InvalidationReport) -> dict[str, object]:
    """The same report as JSON, surfaces and per-record resolutions included."""
    payload = asdict(report)
    payload["scanned"] = report.scanned
    payload["entries"] = list(report.entries)
    payload["unread"] = [reading.surface for reading in report.unread]
    return payload


def render_json(report: InvalidationReport) -> str:
    return json.dumps(report_payload(report), indent=2, sort_keys=True)
