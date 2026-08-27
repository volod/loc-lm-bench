"""Printing the currency report: one row per registered family, plus what each registry answered.

Two shapes of the same reading. The text table is what an operator scans -- carried beside upstream,
verdict in between, and under each row the registries that were asked, what they offered, and when
they answered. The JSON payload is the same content for a script, because "is the roster current"
is a question something other than a human will eventually ask.
"""

import json
from dataclasses import asdict

from llb.backends.currency.report import BEHIND, CURRENT, UNKNOWN, FamilyCurrency

PREFIX = "[currency]"
_HEADER = ("family", "carried", "verdict", "upstream", "read from")
_ORDER = (BEHIND, UNKNOWN, CURRENT)


def _row_cells(row: FamilyCurrency) -> tuple[str, str, str, str, str]:
    upstream = row.upstream.id if row.upstream else "-"
    return (row.family_id, row.carried or "-", row.verdict, upstream, row.registry or "-")


def _table(rows: tuple[FamilyCurrency, ...]) -> list[str]:
    cells = [_HEADER] + [_row_cells(row) for row in rows]
    widths = [max(len(cell[column]) for cell in cells) for column in range(len(_HEADER))]
    return [
        f"{PREFIX} " + "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(entry))
        for entry in cells
    ]


def _reading_lines(row: FamilyCurrency) -> list[str]:
    lines = []
    for reading in row.readings:
        namespace = reading.namespace or "(none)"
        when = f" at {reading.read_at}" if reading.read_at else ""
        if reading.error:
            lines.append(f"{PREFIX}     {reading.registry} `{namespace}`{when}: {reading.error}")
            continue
        offered = reading.newest
        evidence = f" from `{offered.evidence}`" if offered else ""
        newest_id = offered.id if offered else "-"
        lines.append(
            f"{PREFIX}     {reading.registry} `{namespace}`{when}: newest {newest_id}{evidence}"
        )
    return lines


def counts(rows: tuple[FamilyCurrency, ...]) -> dict[str, int]:
    """How many families landed in each verdict, in report order."""
    return {verdict: sum(1 for row in rows if row.verdict == verdict) for verdict in _ORDER}


def render_text(rows: tuple[FamilyCurrency, ...], *, detail: bool = True) -> str:
    """The operator-facing report: the table, each family's registry readings, and the tally."""
    lines = _table(rows)
    if detail:
        detailed: list[str] = [lines[0]]
        for line, row in zip(lines[1:], rows):
            detailed.append(line)
            if row.reason:
                detailed.append(f"{PREFIX}     reason: {row.reason}")
            detailed.extend(_reading_lines(row))
        lines = detailed
    tally = ", ".join(f"{count} {verdict}" for verdict, count in counts(rows).items())
    lines.append(f"{PREFIX} {len(rows)} families: {tally}")
    return "\n".join(lines)


def report_payload(rows: tuple[FamilyCurrency, ...]) -> dict[str, object]:
    """The same report as JSON, readings and read times included."""
    return {
        "families": [asdict(row) for row in rows],
        "counts": counts(rows),
        "verdicts": [CURRENT, BEHIND, UNKNOWN],
    }


def render_json(rows: tuple[FamilyCurrency, ...]) -> str:
    return json.dumps(report_payload(rows), indent=2, sort_keys=True)
