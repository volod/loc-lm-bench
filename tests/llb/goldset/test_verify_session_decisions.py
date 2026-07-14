"""Tests for verify session decisions."""

import json
from llb.goldset.verify_base import load_worksheet, write_worksheet_rows
from llb.goldset.verify_sampling.worksheet import build_sample_worksheet
from llb.goldset.verify_session.loop import run_session
from llb.goldset.verify_session.report import (
    SESSION_STATS_FILENAME,
    SessionStats,
    throughput_line,
)
from tests.llb.goldset._verify_helpers import (
    _bundle,
    _item,
    _ticking_clock,
    _ws,
    _ws_row,
)


def test_session_reject_infers_code_from_failed_check(tmp_path):
    path = _ws(tmp_path, [_ws_row("a")])
    out: list[str] = []
    run_session(path, inputs=iter(["R", "x", "q"]), output=out.append)
    rows, _ = load_worksheet(path)
    assert rows[0]["decision"] == "reject" and rows[0]["reject_code"] == "wrong_reference"
    assert any("inferred" in line for line in out)


def test_session_reject_explicit_code_and_invalid_code(tmp_path):
    path = _ws(tmp_path, [_ws_row("a"), _ws_row("b")])
    out: list[str] = []
    run_session(path, inputs=iter(["x bad_question", "x nonsense", "q"]), output=out.append)
    rows, _ = load_worksheet(path)
    by_id = {r["item_id"]: r for r in rows}
    assert by_id["a"]["reject_code"] == "bad_question"
    assert by_id["b"]["decision"] == ""  # an unknown code refuses to decide
    assert any("unknown reject code" in line for line in out)


def test_session_accept_clears_stale_reject_code(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", "reject", reject_code="bad_question")])
    run_session(path, inputs=iter(["j1", "y", "q"]), output=[].append)
    rows, _ = load_worksheet(path)
    assert rows[0]["decision"] == "accept" and rows[0]["reject_code"] == ""


def test_session_edit_regrounds_immediately(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    ws = bundle / "verify_sample.csv"
    build_sample_worksheet(bundle, ws, n=1)  # manifest beside ws resolves the corpus root
    out: list[str] = []
    run_session(ws, inputs=iter(["e", "Новограді-Волинському", "y", "q"]), output=out.append)
    rows, _ = load_worksheet(ws)
    assert rows[0]["edited_answer"] == "Новограді-Волинському"
    assert rows[0]["decision"] == "accept"
    assert any("re-grounded" in line for line in out)


def test_session_edit_blocked_when_not_verbatim(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    ws = bundle / "verify_sample.csv"
    build_sample_worksheet(bundle, ws, n=1)
    out: list[str] = []
    run_session(ws, inputs=iter(["e", "цього немає в корпусі", "q"]), output=out.append)
    rows, _ = load_worksheet(ws)
    assert rows[0]["edited_answer"] == ""  # the un-groundable edit was refused on the spot
    assert any("BLOCKED" in line for line in out)


def test_session_accept_blocked_until_stale_edit_regrounds(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    ws = bundle / "verify_sample.csv"
    build_sample_worksheet(bundle, ws, n=1)
    rows, fields = load_worksheet(ws)
    rows[0]["edited_answer"] = "рядок не з корпусу"  # simulate a hand-edited CSV cell
    write_worksheet_rows(ws, rows, fields)
    out: list[str] = []
    run_session(ws, inputs=iter(["y", "q"]), output=out.append)
    rows, _ = load_worksheet(ws)
    assert rows[0]["decision"] == ""  # accept refused until the edit re-grounds
    assert any("BLOCKED" in line for line in out)


def test_session_stats_measure_items_per_hour(tmp_path):
    path = _ws(tmp_path, [_ws_row("a"), _ws_row("b")])
    out: list[str] = []
    decided = run_session(
        path, inputs=iter(["y", "y", "q"]), output=out.append, clock=_ticking_clock()
    )
    assert decided == 2
    assert any("items/h" in line for line in out)  # per-decision pace + end-of-session summary
    stats = json.loads((path.with_name(SESSION_STATS_FILENAME)).read_text(encoding="utf-8"))
    record = stats["sessions"][-1]
    assert record["decided_this_session"] == 2
    assert record["items_per_hour"] > 0
    assert record["total_rows"] == 2


def test_session_without_decisions_writes_no_stats(tmp_path):
    path = _ws(tmp_path, [_ws_row("a")])
    run_session(path, inputs=iter(["n", "q"]), output=[].append, clock=_ticking_clock())
    assert not path.with_name(SESSION_STATS_FILENAME).exists()


def test_throughput_line_reports_rate_and_eta():
    clock = _ticking_clock(step=60.0)
    stats = SessionStats(clock=clock)  # started at 60
    stats.on_decision()
    rows = [_ws_row("a", "accept"), _ws_row("b", ""), _ws_row("c", "")]
    line = throughput_line(stats, rows)  # elapsed 60s, 1 decision -> 60 items/h, 2 remaining
    assert "60.0 items/h" in line and "2 remaining" in line
