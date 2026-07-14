"""Tests for rate session."""

from llb.judge.calibration_worksheet import load_worksheet
from llb.judge.rate.presentation import completion_panel
from llb.judge.rate.session import _go_forward, _go_unrated
from llb.judge.rate.state import (
    advanced_index,
    save_human_columns,
)
from test_rate import _by_id, _make_ws, _row, _run_session


def test_session_author_rate_navigate_quit(tmp_path):
    # author an answer on item1, rate it 4, go back and re-rate 3, jump to item3, rate 2, quit.
    result = _run_session(
        tmp_path,
        [_row("a"), _row("b"), _row("c")],
        ["a", "Kyiv is the capital", "4", "p", "3", "j 3", "2", "q"],
    )

    assert result.rated == 2
    by_id = _by_id(result.rows)
    assert by_id["a"]["human_answer"] == "Kyiv is the capital"
    assert by_id["a"]["human_rating"] == "3" and by_id["a"]["human_status"] == "rated"
    assert by_id["b"]["human_rating"] == ""  # never rated
    assert by_id["c"]["human_rating"] == "2"


def test_session_resume_starts_at_first_unrated(tmp_path):
    result = _run_session(tmp_path, [_row("a", human_rating="5"), _row("b"), _row("c")], ["q"])
    assert any("item 2/3" in line for line in result.output)  # resumed at first unrated (item b)


def test_session_start_option_overrides_resume(tmp_path):
    result = _run_session(tmp_path, [_row("a"), _row("b"), _row("c")], ["q"], start=3)
    assert any("item 3/3" in line for line in result.output)


def test_session_clear_command_resets_rating(tmp_path):
    # A fully-rated worksheet opens on the completion screen, so navigate to the item (p) first.
    result = _run_session(
        tmp_path, [_row("a", human_rating="4", human_status="rated")], ["p", "c", "q"]
    )
    assert result.rows[0]["human_rating"] == "" and result.rows[0]["human_status"] == "pending"


def test_advanced_index_transitions():
    rated = [_row("a", human_rating="5"), _row("b", human_rating="5")]
    assert advanced_index(0, 2, rated) == 1  # not last -> next item
    assert advanced_index(1, 2, rated) == 2  # last + all rated -> completion screen (==total)
    gap = [_row("a"), _row("b", human_rating="5")]  # index 0 still unrated
    assert advanced_index(1, 2, gap) == 0  # last + gap -> wrap to the first unrated


def test_go_forward_returns_next_index_and_reports_gap():
    output: list[str] = []
    rows = [_row("a"), _row("b", human_rating="5")]
    assert _go_forward(1, 2, rows, output.append) == 0
    assert any("1 item(s) still unrated" in line for line in output)


def test_go_unrated_handles_completion_index():
    output: list[str] = []
    rated = [_row("a", human_rating="5")]
    assert _go_unrated(len(rated), rated, output.append) == len(rated)
    assert any("all items are rated" in line for line in output)

    gap = [_row("a", human_rating="5"), _row("b")]
    assert _go_unrated(len(gap), gap, output.append) == 1


def test_completion_panel_reports_spread_and_finish():
    rows = [_row("a", human_rating="5", human_answer="x"), _row("b", human_rating="4")]
    panel = completion_panel(rows, 2)
    assert "all 2 items rated" in panel and "1 with your own answer" in panel
    assert "5:1" in panel and "4:1" in panel
    assert "finish" in panel.lower()


def test_session_lands_on_completion_after_rating_last(tmp_path):
    # rate both; after the last, all rated -> completion screen; Enter finishes.
    result = _run_session(tmp_path, [_row("a"), _row("b")], ["3", "4", ""])
    assert result.rated == 2
    assert any("all 2 items rated" in line for line in result.output)


def test_session_opens_on_completion_when_all_already_rated(tmp_path):
    result = _run_session(
        tmp_path, [_row("a", human_rating="5"), _row("b", human_rating="4")], ["q"]
    )
    assert any("all 2 items rated" in line for line in result.output)


def test_session_completion_unrated_reports_all_rated(tmp_path):
    result = _run_session(tmp_path, [_row("a", human_rating="5")], ["u", "q"])
    assert any("all items are rated" in line for line in result.output)


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
    # open on completion -> p goes to the last item -> clear it -> q.
    result = _run_session(
        tmp_path, [_row("a", human_rating="5"), _row("b", human_rating="4")], ["p", "c", "q"]
    )
    assert result.rows[1]["human_rating"] == ""  # changed via the review screen


def test_session_clear_flag_confirmed(tmp_path):
    result = _run_session(
        tmp_path,
        [_row("a", human_rating="4"), _row("b", human_rating="5")],
        ["yes", "q"],
        clear=True,
    )
    assert result.rated == 0
    assert all(row["human_rating"] == "" for row in result.rows)


def test_session_clear_flag_aborted_keeps_data(tmp_path):
    result = _run_session(tmp_path, [_row("a", human_rating="4")], ["no"], clear=True)
    assert result.rated == 1
    assert result.rows[0]["human_rating"] == "4"
    assert any("clear aborted" in line for line in result.output)
    assert not any("item 1/1" in line for line in result.output)


def test_session_keyboardinterrupt_still_saves(tmp_path):
    def feed():
        yield "4"  # rate item a -> 4 (written through)
        raise KeyboardInterrupt  # Ctrl-C while at item b

    result = _run_session(tmp_path, [_row("a"), _row("b")], feed())
    assert result.rated == 1
    assert result.rows[0]["human_rating"] == "4"


def test_session_eof_saves_and_returns(tmp_path):
    # exhausting the injected inputs (no 'q') is treated as save + quit.
    result = _run_session(tmp_path, [_row("a")], ["4"])
    assert result.rated == 1
    assert result.rows[0]["human_rating"] == "4"
