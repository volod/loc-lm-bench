"""Worksheet rating state, progress arithmetic, and persistence."""

import csv
from collections.abc import Sequence
from pathlib import Path

from llb.judge.calibration import HUMAN_COLS, load_worksheet, write_worksheet_rows
from llb.judge.rate.commands import RATING_MAX, RATING_MIN

STATUS_PENDING = "pending"
STATUS_RATED = "rated"


def first_unrated_index(rows: Sequence[dict[str, str]]) -> int:
    """Return the first unrated row index, or zero when all rows are rated."""
    for index, row in enumerate(rows):
        if not (row.get("human_rating") or "").strip():
            return index
    return 0


def rated_count(rows: Sequence[dict[str, str]]) -> int:
    return sum(1 for row in rows if (row.get("human_rating") or "").strip())


def rating_histogram(rows: Sequence[dict[str, str]]) -> dict[int, int]:
    histogram = {rating: 0 for rating in range(RATING_MIN, RATING_MAX + 1)}
    for row in rows:
        value = (row.get("human_rating") or "").strip()
        if value.isdigit() and int(value) in histogram:
            histogram[int(value)] += 1
    return histogram


def advanced_index(index: int, total: int, rows: Sequence[dict[str, str]]) -> int:
    """Advance normally, or wrap from the last row to the first unrated gap."""
    if index < total - 1:
        return index + 1
    if rated_count(rows) == total:
        return total
    return first_unrated_index(rows)


def clear_human_columns(rows: Sequence[dict[str, str]]) -> None:
    for row in rows:
        for column in HUMAN_COLS:
            row[column] = ""


def save_human_columns(
    path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]
) -> None:
    """Merge only human-owned columns into the current on-disk worksheet."""
    try:
        disk_rows, disk_fields = load_worksheet(path)
    except (OSError, csv.Error):
        disk_rows = []
        disk_fields = list(fieldnames)
    if not disk_rows:
        write_worksheet_rows(path, rows, fieldnames)
        return
    human_by_id = {
        row["item_id"]: {column: row.get(column, "") for column in HUMAN_COLS}
        for row in rows
        if row.get("item_id")
    }
    for disk_row in disk_rows:
        overlay = human_by_id.get(disk_row.get("item_id", ""))
        if overlay is not None:
            disk_row.update(overlay)
    write_worksheet_rows(path, disk_rows, disk_fields)


def set_rating(row: dict[str, str], value: int) -> None:
    row["human_rating"] = str(value)
    row["human_status"] = STATUS_RATED


def clear_rating(row: dict[str, str]) -> None:
    row["human_rating"] = ""
    row["human_status"] = STATUS_PENDING
