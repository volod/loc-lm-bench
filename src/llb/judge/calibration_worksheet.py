"""Focused calibration worksheet implementation."""

import csv
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from llb.core.contracts.judging import WorksheetItem
from llb.core.fsutil import atomic_write_text
from llb.goldset.schema import GoldItem

_LOG = logging.getLogger(__name__)

HUMAN_COLS = ["human_answer", "human_rating", "human_note", "human_status"]

WORKSHEET_COLS = [
    "item_id",
    "split",
    "provenance",
    "question",
    "reference_answer",
    "model_answer",
    "human_answer",
    "human_rating",
    "human_note",
    "human_status",
    "judge_rating",
]


def worksheet_fieldnames(existing: Sequence[str] | None = None) -> list[str]:
    """Canonical column order: keep any columns already in the header, then append any
    `WORKSHEET_COLS` that are missing. With no header yet, the order is `WORKSHEET_COLS`."""
    names = list(existing) if existing else []
    for col in WORKSHEET_COLS:
        if col not in names:
            names.append(col)
    return names


def load_worksheet(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load a worksheet CSV into `(rows, fieldnames)`.

    Any `WORKSHEET_COLS` column missing from the header is added blank, so callers can rely on
    every column being present; any extra columns are preserved in `fieldnames` so a round-trip
    never drops data.
    """
    text = Path(path).read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    fieldnames = worksheet_fieldnames(reader.fieldnames)
    rows = [{name: (raw.get(name) or "") for name in fieldnames} for raw in reader]
    return rows, fieldnames


def write_worksheet_rows(
    out_path: Path,
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str] | None = None,
) -> int:
    """Atomically (re)write the whole worksheet, preserving column order.

    The CSV is the worksheet's only state, so every edit rewrites it through a temp file +
    `os.replace` (`atomic_write_text`); a crash mid-write leaves the prior file intact.
    """
    columns = list(fieldnames) if fieldnames else list(WORKSHEET_COLS)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in columns})
    atomic_write_text(Path(out_path), buf.getvalue())
    return len(rows)


def emit_worksheet(items: list[WorksheetItem], out_path: Path) -> int:
    """Write a blank CSV worksheet (one row per calibration item) for the human to fill.

    `model_answer`, the human columns, and `judge_rating` are blank; `provenance` is copied
    from the item. Use `write_filled_worksheet` instead to pre-fill `model_answer` from a run.
    """
    rows = [
        {
            "item_id": it["id"],
            "split": it["split"],
            "provenance": it.get("provenance", "") or "",
            "question": it["question"],
            "reference_answer": it["reference_answer"],
        }
        for it in items
        if it.get("split") == "calibration"
    ]
    return write_worksheet_rows(out_path, rows)


def _existing_rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    """Index a prior worksheet by `item_id` for the merge-on-regenerate path (empty if none)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        rows, _ = load_worksheet(path)
    except (OSError, csv.Error):
        return {}
    return {row["item_id"]: row for row in rows if row.get("item_id")}


def _merge_human_columns(new_row: dict[str, str], prev_row: dict[str, str]) -> None:
    """Carry a prior run's human columns into a freshly pre-filled row.

    Human work survives a re-run with the same deterministic candidate. If the regenerated
    `model_answer` CHANGED (a different candidate), the human rating no longer applies to the
    shown answer, so it is cleared with a warning; the human's OWN authored answer/note are
    kept (they do not depend on the candidate).
    """
    for col in HUMAN_COLS:
        prev_val = prev_row.get(col, "")
        if prev_val:
            new_row[col] = prev_val
    answer_changed = prev_row.get("model_answer", "") != new_row.get("model_answer", "")
    if answer_changed and new_row.get("human_rating"):
        _LOG.warning(
            "[calibration] item %s: model_answer changed since the last rating; clearing the "
            "stale human_rating (was %r) -- re-rate against the new answer.",
            new_row.get("item_id", "?"),
            new_row["human_rating"],
        )
        new_row["human_rating"] = ""
        if new_row.get("human_status") == "rated":
            new_row["human_status"] = "pending"


def write_filled_worksheet(
    answers: Sequence[tuple[GoldItem, str]],
    out_path: Path,
    judge_ratings: Sequence[float] | None = None,
) -> int:
    """Write a worksheet with model_answer pre-filled from a run; human columns left blank.

    `answers` is a list of `(gold_item, model_answer)` (gold_item duck-typed:
    `id / split / question / reference_answer / provenance`). Produced by `run-eval --worksheet`
    on the calibration split so the human authors `human_answer` + `human_rating`.

    When `judge_ratings` is supplied (aligned with `answers`), the `judge_rating` column is
    pre-filled with the JUDGE's score per item -- so the calibration worksheet carries both the
    judge rating and a blank human rating, and `calibration score` can compute rho(human, judge)
    once the human column is filled.

    Re-running MERGES any prior human columns by `item_id` (never clobbers them); a row whose
    regenerated `model_answer` changed has its stale rating cleared (see `_merge_human_columns`).
    """
    existing = _existing_rows_by_id(out_path)
    rows: list[dict[str, str]] = []
    for i, (item, answer) in enumerate(answers):
        judge = "" if judge_ratings is None else str(round(float(judge_ratings[i]), 4))
        row = {
            "item_id": item.id,
            "split": item.split,
            "provenance": getattr(item, "provenance", "") or "",
            "question": item.question,
            "reference_answer": item.reference_answer,
            "model_answer": answer or "",
            "human_answer": "",
            "human_rating": "",
            "human_note": "",
            "human_status": "",
            "judge_rating": judge,
        }
        prev = existing.get(str(item.id))
        if prev is not None:
            _merge_human_columns(row, prev)
        rows.append(row)
    return write_worksheet_rows(out_path, rows)
