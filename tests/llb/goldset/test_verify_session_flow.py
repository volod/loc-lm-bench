"""Tests for verify session flow."""

from llb.goldset.verify_base import load_worksheet
from llb.goldset.verify_session.commands import _go_forward, _go_undecided
from llb.goldset.verify_session.loop import run_session
from llb.goldset.verify_session.report import (
    decided_count,
    first_undecided_index,
)
from tests.llb.goldset._verify_helpers import (
    _ws,
    _ws_row,
)


def test_session_marks_checks_and_decides(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", stratum="s", synthetic="false"), _ws_row("b", stratum="s")])
    out: list[str] = []
    # item a: grounded pass, reference fail, accept -> advances; item b: reject; then finish.
    decided = run_session(
        path,
        inputs=iter(["g", "R", "y", "x", "q"]),
        output=out.append,
        show_crosscheck=False,
    )
    assert decided == 2
    rows, _ = load_worksheet(path)
    by_id = {r["item_id"]: r for r in rows}
    assert by_id["a"]["chk_grounded"] == "pass" and by_id["a"]["chk_reference"] == "fail"
    assert by_id["a"]["decision"] == "accept" and by_id["b"]["decision"] == "reject"


def test_session_resumes_at_first_undecided(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", "accept"), _ws_row("b", ""), _ws_row("c", "")])
    assert first_undecided_index(load_worksheet(path)[0]) == 1
    out: list[str] = []
    run_session(path, inputs=iter(["y", "q"]), output=out.append)
    rows, _ = load_worksheet(path)
    # The session opened on item b (first undecided) and accepted it; a stayed accepted, c untouched.
    assert {r["item_id"]: r["decision"] for r in rows} == {"a": "accept", "b": "accept", "c": ""}


def test_session_planted_check_rejected_for_real_item(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", synthetic="false")])
    out: list[str] = []
    run_session(path, inputs=iter(["p", "q"]), output=out.append)
    rows, _ = load_worksheet(path)
    assert rows[0]["chk_planted"] == ""  # the N/A planted mark was refused
    assert any("N/A" in line for line in out)


def test_session_save_preserves_context_column(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", context="some>>>span<<<text")])
    run_session(path, inputs=iter(["y", "q"]), output=[].append)
    rows, _ = load_worksheet(path)
    assert rows[0]["context"] == "some>>>span<<<text"  # sampler-owned column not clobbered


def test_session_clear_flag_confirmed(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", "accept"), _ws_row("b", "reject")])
    decided = run_session(path, inputs=iter(["yes", "q"]), output=[].append, clear=True)
    rows, _ = load_worksheet(path)
    assert decided == 0
    assert all(row["decision"] == "" for row in rows)


def test_session_clear_flag_aborted_keeps_data(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", "accept")])
    out: list[str] = []
    decided = run_session(path, inputs=iter(["no"]), output=out.append, clear=True)
    rows, _ = load_worksheet(path)
    assert decided == 1
    assert rows[0]["decision"] == "accept"
    assert any("clear aborted" in line for line in out)
    assert not any("item 1/1" in line for line in out)


def test_session_completion_undecided_reports_all_decided(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", "accept")])
    out: list[str] = []
    run_session(path, inputs=iter(["u", "q"]), output=out.append)
    assert any("all items are decided" in line for line in out)


def test_go_forward_returns_next_index_and_reports_gap():
    output: list[str] = []
    rows = [_ws_row("a", ""), _ws_row("b", "accept")]
    assert _go_forward(1, 2, rows, output.append) == 0
    assert any("1 item(s) still undecided" in line for line in output)


def test_go_undecided_handles_completion_index():
    output: list[str] = []
    decided = [_ws_row("a", "accept")]
    assert _go_undecided(len(decided), decided, output.append) == len(decided)
    assert any("all items are decided" in line for line in output)

    gap = [_ws_row("a", "accept"), _ws_row("b", "")]
    assert _go_undecided(len(gap), gap, output.append) == 1


def test_decided_count(tmp_path):
    rows = [_ws_row("a", "accept"), _ws_row("b", "reject"), _ws_row("c", "")]
    assert decided_count(rows) == 2


def test_session_confidence_order_reviews_suspicious_first(tmp_path):
    path = _ws(
        tmp_path,
        [_ws_row("good", cc_grounded="true"), _ws_row("bad", cc_grounded="false")],
    )
    run_session(path, inputs=iter(["y", "q"]), output=[].append, order="confidence")
    rows, _ = load_worksheet(path)
    # The single accept landed on the LOW-confidence row, and the CSV order never changed.
    assert [r["item_id"] for r in rows] == ["good", "bad"]
    assert {r["item_id"]: r["decision"] for r in rows} == {"good": "", "bad": "accept"}
