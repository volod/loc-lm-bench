"""Pure worksheet-row summaries and the session throughput stats for the human verification gate.

No terminal I/O here: progress/tally helpers, the completion + end-of-session report, the atomic
human-column merge-save, and the measured items-per-hour `SessionStats` (clock injected so tests
never sleep). The interactive session (`commands`, `decision`, `loop`) builds on these.
"""

import csv
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.verify import (
    ACCEPT,
    HUMAN_COLS,
    REJECT,
    load_worksheet,
    write_worksheet_rows,
)

SESSION_STATS_FILENAME = "verify_session_stats.json"


def first_undecided_index(rows: Sequence[dict[str, str]]) -> int:
    """Index of the first row without an accept/reject decision (resume point). 0 if all decided."""
    for i, row in enumerate(rows):
        if (row.get("decision") or "").strip() not in (ACCEPT, REJECT):
            return i
    return 0


def decided_count(rows: Sequence[dict[str, str]]) -> int:
    """How many rows carry an accept/reject decision."""
    return sum(1 for row in rows if (row.get("decision") or "").strip() in (ACCEPT, REJECT))


def decision_tally(rows: Sequence[dict[str, str]]) -> tuple[int, int]:
    """`(accepted, rejected)` counts over decided rows."""
    accepted = sum(1 for row in rows if (row.get("decision") or "").strip() == ACCEPT)
    rejected = sum(1 for row in rows if (row.get("decision") or "").strip() == REJECT)
    return accepted, rejected


def _advanced_index(idx: int, total: int, rows: Sequence[dict[str, str]]) -> int:
    """Where to go after deciding the item at `idx`: next item, else wrap to the first undecided,
    else the completion screen once everything is decided (never stuck re-showing the last card)."""
    if idx < total - 1:
        return idx + 1
    if decided_count(rows) == total:
        return total
    return first_undecided_index(rows)


def completion_panel(rows: Sequence[dict[str, str]], total: int) -> str:
    """The 'all items decided' review screen shown once you advance past the last item."""
    accepted, rejected = decision_tally(rows)
    return "\n".join(
        [
            f"===== all {total} items decided (accept={accepted}, reject={rejected}) =====",
            "  review/change: b = last item, j <N> = jump to item N, u = next undecided",
            "  finish: press Enter or q to save + quit (then run make verify-accept)",
        ]
    )


def clear_human_columns(rows: Sequence[dict[str, str]]) -> None:
    """Wipe every human column in place (the `--clear` start-fresh path)."""
    for row in rows:
        for col in HUMAN_COLS:
            row[col] = ""


def save_human_columns(
    path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]
) -> None:
    """Persist ONLY the human columns, merged into the CURRENT on-disk worksheet by `item_id`.

    Re-reading the file on each save and overlaying only the human columns means a context column
    the sampler owns is never clobbered by the session's load-time snapshot. Falls back to a full
    write if the file is missing or unreadable.
    """
    try:
        disk_rows, disk_fields = load_worksheet(path)
    except (OSError, csv.Error):
        write_worksheet_rows(path, rows, fieldnames)
        return
    if not disk_rows:
        write_worksheet_rows(path, rows, fieldnames)
        return
    human_by_id = {
        row["item_id"]: {col: row.get(col, "") for col in HUMAN_COLS}
        for row in rows
        if row.get("item_id")
    }
    for disk_row in disk_rows:
        overlay = human_by_id.get(disk_row.get("item_id", ""))
        if overlay is not None:
            disk_row.update(overlay)
    write_worksheet_rows(path, disk_rows, disk_fields)


@dataclass
class SessionStats:
    """Wall-clock throughput of ONE review sitting.

    Decisions per elapsed hour is the measured reviewer-throughput number the human evidence
    records; the clock is injected so tests never sleep.
    """

    clock: Callable[[], float]
    started: float = 0.0
    decisions: int = 0

    def __post_init__(self) -> None:
        self.started = self.clock()

    def on_decision(self) -> None:
        self.decisions += 1

    def elapsed_seconds(self) -> float:
        return max(self.clock() - self.started, 0.0)

    def items_per_hour(self) -> float:
        elapsed = self.elapsed_seconds()
        if not self.decisions or elapsed <= 0:
            return 0.0
        return self.decisions * 3600.0 / elapsed


def summary_lines(
    rows: Sequence[dict[str, str]], path: Path, stats: "SessionStats | None" = None
) -> list[str]:
    """The end-of-session report: progress, accept/reject split, pace, and the next command."""
    total = len(rows)
    decided = decided_count(rows)
    accepted, rejected = decision_tally(rows)
    lines = [
        f"[verify] saved {path}",
        f"[verify] progress : {decided}/{total} decided, {total - decided} remaining "
        f"(accept {accepted}, reject {rejected})",
    ]
    if stats is not None and stats.decisions:
        lines.append(
            f"[verify] pace     : {stats.decisions} decided this session in "
            f"{stats.elapsed_seconds() / 60.0:.1f} min -- {stats.items_per_hour():.1f} items/h "
            f"(recorded in {SESSION_STATS_FILENAME})"
        )
    if decided < total:
        lines.append(
            "[verify] resume   : re-run `make verify-review` (continues at the first undecided item)"
        )
    lines.append(f"[verify] accept   : make verify-accept VERIFY_WS={path} BUNDLE=<draft bundle>")
    return lines


def throughput_line(stats: SessionStats, rows: Sequence[dict[str, str]]) -> str:
    """One-line session pace: decided count, items/hour, and the ETA for the remaining rows."""
    remaining = len(rows) - decided_count(rows)
    rate = stats.items_per_hour()
    minutes = stats.elapsed_seconds() / 60.0
    line = f"[stats] session: {stats.decisions} decided in {minutes:.1f} min"
    if rate > 0:
        line += f" -- {rate:.1f} items/h"
        if remaining:
            line += f"; ~{remaining * 60.0 / rate:.0f} min for {remaining} remaining"
    return line


def append_session_stats(worksheet_path: Path, record: dict[str, object]) -> Path:
    """Append one session record to `verify_session_stats.json` beside the worksheet.

    The durable trace of measured reviewer throughput (what the current docs cite), so a
    finished 40-item pass does not live only in scrollback.
    """
    path = Path(worksheet_path).with_name(SESSION_STATS_FILENAME)
    payload: dict[str, object] = {"sessions": []}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("sessions"), list):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            pass
    sessions = payload["sessions"]
    assert isinstance(sessions, list)
    sessions.append(record)
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def _session_record(stats: SessionStats, rows: Sequence[dict[str, str]]) -> dict[str, object]:
    return {
        "decided_this_session": stats.decisions,
        "elapsed_seconds": round(stats.elapsed_seconds(), 1),
        "items_per_hour": round(stats.items_per_hour(), 1),
        "total_decided": decided_count(rows),
        "total_rows": len(rows),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
