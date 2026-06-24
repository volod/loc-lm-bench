"""Tests for the interactive calibration rater (`llb.judge.rate`).

The pure pieces (parse_command, first_unrated_index, format_card, clear) are checked
directly; the session loop is driven by an INJECTED input iterator + output sink, so no
terminal / model / endpoint / GPU is needed -- it operates only on the CSV.
"""

from llb.judge.calibration import WORKSHEET_COLS, load_worksheet, write_worksheet_rows
from llb.judge.rate import (
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
    _advanced_index,
    clear_human_columns,
    completion_panel,
    first_unrated_index,
    format_card,
    parse_command,
    rating_histogram,
    run_session,
    save_human_columns,
    summary_lines,
)


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


def test_session_author_rate_navigate_quit(tmp_path):
    path = _make_ws(tmp_path, [_row("a"), _row("b"), _row("c")])
    out: list[str] = []
    # author an answer on item1, rate it 4, go back and re-rate 3, jump to item3, rate 2, quit.
    inputs = ["a", "Kyiv is the capital", "4", "p", "3", "j 3", "2", "q"]
    rated = run_session(path, inputs=inputs, output=out.append)

    assert rated == 2
    rows, _ = load_worksheet(path)
    by_id = {r["item_id"]: r for r in rows}
    assert by_id["a"]["human_answer"] == "Kyiv is the capital"
    assert by_id["a"]["human_rating"] == "3" and by_id["a"]["human_status"] == "rated"
    assert by_id["b"]["human_rating"] == ""  # never rated
    assert by_id["c"]["human_rating"] == "2"


def test_session_resume_starts_at_first_unrated(tmp_path):
    path = _make_ws(tmp_path, [_row("a", human_rating="5"), _row("b"), _row("c")])
    out: list[str] = []
    run_session(path, inputs=["q"], output=out.append)
    assert any("item 2/3" in line for line in out)  # resumed at first unrated (item b)


def test_session_start_option_overrides_resume(tmp_path):
    path = _make_ws(tmp_path, [_row("a"), _row("b"), _row("c")])
    out: list[str] = []
    run_session(path, inputs=["q"], output=out.append, start=3)
    assert any("item 3/3" in line for line in out)


def test_session_clear_command_resets_rating(tmp_path):
    # A fully-rated worksheet opens on the completion screen, so navigate to the item (p) first.
    path = _make_ws(tmp_path, [_row("a", human_rating="4", human_status="rated")])
    run_session(path, inputs=["p", "c", "q"], output=lambda _s: None)
    rows, _ = load_worksheet(path)
    assert rows[0]["human_rating"] == "" and rows[0]["human_status"] == "pending"


def test_advanced_index_transitions():
    rated = [_row("a", human_rating="5"), _row("b", human_rating="5")]
    assert _advanced_index(0, 2, rated) == 1  # not last -> next item
    assert _advanced_index(1, 2, rated) == 2  # last + all rated -> completion screen (==total)
    gap = [_row("a"), _row("b", human_rating="5")]  # index 0 still unrated
    assert _advanced_index(1, 2, gap) == 0  # last + gap -> wrap to the first unrated


def test_completion_panel_reports_spread_and_finish():
    rows = [_row("a", human_rating="5", human_answer="x"), _row("b", human_rating="4")]
    panel = completion_panel(rows, 2)
    assert "all 2 items rated" in panel and "1 with your own answer" in panel
    assert "5:1" in panel and "4:1" in panel
    assert "finish" in panel.lower()


def test_session_lands_on_completion_after_rating_last(tmp_path):
    path = _make_ws(tmp_path, [_row("a"), _row("b")])
    out: list[str] = []
    # rate both; after the last, all rated -> completion screen; Enter finishes.
    rated = run_session(path, inputs=["3", "4", ""], output=out.append)
    assert rated == 2
    assert any("all 2 items rated" in line for line in out)


def test_session_opens_on_completion_when_all_already_rated(tmp_path):
    path = _make_ws(tmp_path, [_row("a", human_rating="5"), _row("b", human_rating="4")])
    out: list[str] = []
    run_session(path, inputs=["q"], output=out.append)
    assert any("all 2 items rated" in line for line in out)


def test_save_human_columns_preserves_disk_non_human_columns(tmp_path):
    # Disk has judge_rating filled (e.g. a calibration-run ran). A stale in-memory snapshot with
    # judge blank must NOT clobber the disk judge column -- only the human columns are overlaid.
    path = _make_ws(tmp_path, [_row("a", judge_rating="0.9"), _row("b", judge_rating="0.4")])
    rows, fields = load_worksheet(path)
    stale = [{**r, "judge_rating": "", "human_rating": "5"} for r in rows]
    save_human_columns(path, stale, fields)
    out, _ = load_worksheet(path)
    assert [r["judge_rating"] for r in out] == ["0.9", "0.4"]  # disk judge preserved
    assert [r["human_rating"] for r in out] == ["5", "5"]  # human columns overlaid


def test_session_completion_review_then_change(tmp_path):
    path = _make_ws(tmp_path, [_row("a", human_rating="5"), _row("b", human_rating="4")])
    # open on completion -> p goes to the last item -> clear it -> q.
    run_session(path, inputs=["p", "c", "q"], output=lambda _s: None)
    rows, _ = load_worksheet(path)
    assert rows[1]["human_rating"] == ""  # changed via the review screen


def test_session_clear_flag_confirmed(tmp_path):
    path = _make_ws(tmp_path, [_row("a", human_rating="4"), _row("b", human_rating="5")])
    rated = run_session(path, inputs=["yes", "q"], output=lambda _s: None, clear=True)
    assert rated == 0
    rows, _ = load_worksheet(path)
    assert all(r["human_rating"] == "" for r in rows)


def test_session_clear_flag_aborted_keeps_data(tmp_path):
    path = _make_ws(tmp_path, [_row("a", human_rating="4")])
    rated = run_session(path, inputs=["no"], output=lambda _s: None, clear=True)
    assert rated == 1
    rows, _ = load_worksheet(path)
    assert rows[0]["human_rating"] == "4"


def test_session_keyboardinterrupt_still_saves(tmp_path):
    path = _make_ws(tmp_path, [_row("a"), _row("b")])

    def feed():
        yield "4"  # rate item a -> 4 (written through)
        raise KeyboardInterrupt  # Ctrl-C while at item b

    rated = run_session(path, inputs=feed(), output=lambda _s: None)
    assert rated == 1
    rows, _ = load_worksheet(path)
    assert rows[0]["human_rating"] == "4"


def test_session_eof_saves_and_returns(tmp_path):
    path = _make_ws(tmp_path, [_row("a")])
    # exhausting the injected inputs (no 'q') is treated as save + quit.
    rated = run_session(path, inputs=["4"], output=lambda _s: None)
    assert rated == 1
    rows, _ = load_worksheet(path)
    assert rows[0]["human_rating"] == "4"
