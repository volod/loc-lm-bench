"""Tests for the interactive calibration rater (`llb.judge.rate`).

The pure pieces (parse_command, first_unrated_index, format_card, clear) are checked
directly; the session loop is driven by an INJECTED input iterator + output sink, so no
terminal / model / endpoint / GPU is needed -- it operates only on the CSV.
"""

from dataclasses import dataclass
from pathlib import Path

from llb.judge.calibration_worksheet import WORKSHEET_COLS, load_worksheet, write_worksheet_rows
from llb.judge.rate.commands import (
    ANSWER,
    CLEAR,
    HELP,
    JUMP,
    NEXT,
    NOTE,
    PREV,
    QUIT,
    RATE,
    UNKNOWN,
    UNRATED,
    parse_command,
)
from llb.judge.rate.presentation import format_card, summary_lines
from llb.judge.rate.session import run_session
from llb.judge.rate.state import (
    clear_human_columns,
    first_unrated_index,
    rating_histogram,
)


@dataclass
class SessionResult:
    path: Path
    rated: int
    rows: list[dict[str, str]]
    output: list[str]


def _row(item_id, **over):
    row = {col: "" for col in WORKSHEET_COLS}
    row.update(
        {
            "item_id": item_id,
            "split": "calibration",
            "provenance": "public-reused",
            "question": f"q-{item_id}",
            "reference_answer": f"ref-{item_id}",
            "model_answer": f"model-{item_id}",
        }
    )
    row.update(over)
    return row


def _make_ws(tmp_path, rows):
    path = tmp_path / "ws.csv"
    write_worksheet_rows(path, rows, WORKSHEET_COLS)
    return path


def _run_session(tmp_path, rows, inputs, **kwargs):
    path = _make_ws(tmp_path, rows)
    output: list[str] = []
    rated = run_session(path, inputs=inputs, output=output.append, **kwargs)
    loaded, _ = load_worksheet(path)
    return SessionResult(path=path, rated=rated, rows=loaded, output=output)


def _by_id(rows):
    return {row["item_id"]: row for row in rows}


# --- pure pieces -----------------------------------------------------------------------


def test_parse_command_basics():
    assert parse_command("").kind == NEXT
    assert parse_command("n").kind == NEXT
    assert parse_command("p").kind == PREV
    assert parse_command("b").kind == PREV
    assert parse_command("u").kind == UNRATED
    assert parse_command("c").kind == CLEAR
    assert parse_command("q").kind == QUIT
    assert parse_command("?").kind == HELP
    assert parse_command("h").kind == HELP
    assert parse_command("a").kind == ANSWER
    assert parse_command("note").kind == NOTE


def test_parse_command_rating_in_range():
    cmd = parse_command("4")
    assert cmd.kind == RATE and cmd.value == 4


def test_parse_command_rating_out_of_range_is_unknown():
    assert parse_command("9").kind == UNKNOWN
    assert parse_command("0").kind == UNKNOWN


def test_parse_command_jump():
    assert parse_command("j 5") == parse_command("j5")
    cmd = parse_command("j 5")
    assert cmd.kind == JUMP and cmd.value == 5
    assert parse_command("jx").kind == UNKNOWN


def test_parse_command_arrow_keys():
    assert parse_command("\x1b[A").kind == PREV  # up
    assert parse_command("\x1b[D").kind == PREV  # left
    assert parse_command("\x1b[B").kind == NEXT  # down
    assert parse_command("\x1b[C").kind == NEXT  # right


def test_rating_histogram_counts_and_ignores_invalid():
    rows = [
        _row("a", human_rating="5"),
        _row("b", human_rating="5"),
        _row("c"),  # unrated
        _row("d", human_rating="9"),  # out of range -> ignored
    ]
    hist = rating_histogram(rows)
    assert hist[5] == 2 and hist[1] == 0
    assert sum(hist.values()) == 2


def test_summary_lines_reports_progress_ratings_and_score_cmd(tmp_path):
    rows = [_row("a", human_rating="5", human_answer="x"), _row("b")]
    path = tmp_path / "ws.csv"
    blob = "\n".join(summary_lines(rows, path))
    assert "1/2 rated" in blob and "1 with your own answer" in blob
    assert "5:1" in blob  # rating spread surfaced
    assert "resume" in blob.lower()  # rated < total -> resume hint
    assert f"make calibration-score RATINGS={path}" in blob


def test_first_unrated_index():
    rows = [_row("a", human_rating="3"), _row("b"), _row("c", human_rating="5")]
    assert first_unrated_index(rows) == 1
    assert first_unrated_index([_row("a", human_rating="3")]) == 0  # all rated -> 0


def test_format_card_hides_judge_by_default():
    row = _row("a", judge_rating="0.8", human_rating="4")
    card = format_card(row, 1, 3, 1)
    assert "judge" not in card and "0.8" not in card
    assert "item 1/3 (rated 1, remaining 2)" in card
    shown = format_card(row, 1, 3, 1, show_judge=True)
    assert "judge" in shown and "0.8" in shown


def test_clear_human_columns():
    rows = [_row("a", human_rating="4", human_answer="x", human_note="n", human_status="rated")]
    clear_human_columns(rows)
    assert rows[0]["human_rating"] == "" and rows[0]["human_answer"] == ""
    assert rows[0]["human_note"] == "" and rows[0]["human_status"] == ""


# --- session loop (injected I/O) -------------------------------------------------------
