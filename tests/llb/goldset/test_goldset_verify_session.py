"""Tests for the interactive verification session (`llb.goldset.verify_session` + `verify_card`).

The session loop is driven by an INJECTED input iterator + output sink, so no terminal / model /
endpoint / GPU is needed -- it operates only on the CSV. Card rendering and command parsing are
checked directly. Shared factories live in `_verify_helpers.py`; the pure verify pieces are in
`test_goldset_verify.py`.
"""

import json

from llb.goldset.verify import (
    WORKSHEET_COLS,
    build_sample_worksheet,
    load_worksheet,
    write_worksheet_rows,
)
from llb.goldset.verify_session import (
    ACCEPT_CMD,
    CHECK,
    HELP,
    JUMP,
    NEXT,
    PREV,
    QUIT,
    REJECT_CMD,
    SESSION_STATS_FILENAME,
    Command,
    SessionStats,
    _go_forward,
    _go_undecided,
    decided_count,
    first_undecided_index,
    format_card,
    parse_command,
    run_session,
    throughput_line,
)

from tests.llb.goldset._verify_helpers import (
    _bundle,
    _chain,
    _chain_bundle,
    _item,
    _ticking_clock,
    _ws,
    _ws_row,
)


# --- chain review cards + session ---------------------------------------------------------


def test_chain_review_card_is_dense_and_marks_answer_source_comparison(tmp_path):
    bundle = _chain_bundle(tmp_path, [_chain("c1")])
    out = tmp_path / "verify_sample.csv"
    build_sample_worksheet(bundle, out, n=1)
    rows, _ = load_worksheet(out)
    card = format_card(rows[0], 1, 1, 0)
    assert card.startswith("+" * 64)
    assert "\n\n" not in card
    assert "CHAIN 1/1" in card
    assert "STEP 1/2" in card and "STEP 2/2" in card
    assert "\nQ:" in card and "\nA:" in card and "\nSOURCE:" in card
    assert "compare A with SOURCE" in card
    assert ">>>Alpha керує Beta<<<" in card


def test_chain_review_card_truncates_multiline_text_and_colors_tty_fields(tmp_path):
    bundle = _chain_bundle(tmp_path, [_chain("c1")])
    out = tmp_path / "verify_sample.csv"
    build_sample_worksheet(bundle, out, n=1)
    rows, _ = load_worksheet(out)
    steps = json.loads(rows[0]["chain_steps"])
    steps[0]["question"] = ("довге питання з переносом\n" * 20).strip()
    steps[0]["context"] = ("до " * 80) + ">>>точний доказ<<<" + (" після" * 80)
    rows[0]["chain_steps"] = json.dumps(steps, ensure_ascii=False)

    plain = format_card(rows[0], 1, 1, 0, width=72)
    colored = format_card(rows[0], 1, 1, 0, color=True, width=72)
    assert "..." in plain
    assert ">>>точний доказ<<<" in plain
    assert "\033[" not in plain
    assert "\033[1;36mQ:" in colored
    assert "\033[1;32mA:" in colored
    assert "\033[33mSOURCE:" in colored


def test_chain_session_reuses_navigation_and_blocks_answer_edit(tmp_path):
    bundle = _chain_bundle(tmp_path, [_chain("c1")])
    ws = bundle / "verify_sample.csv"
    build_sample_worksheet(bundle, ws, n=1)
    out: list[str] = []
    run_session(ws, inputs=iter(["w", "new answer", "y", "q"]), output=out.append)
    rows, _ = load_worksheet(ws)
    assert rows[0]["decision"] == "accept"
    assert rows[0]["edited_answer"] == ""
    assert any("chain answer edits are not supported" in line for line in out)


# --- parse_command ------------------------------------------------------------------------


def test_parse_check_pass_and_fail():
    assert parse_command("g") == Command(CHECK, field="chk_grounded", value=True)
    assert parse_command("R").kind == CHECK and parse_command("R").value is False


def test_parse_decisions_and_nav():
    assert parse_command("y").kind == ACCEPT_CMD
    assert parse_command("x").kind == REJECT_CMD
    assert parse_command("").kind == NEXT
    assert parse_command("b").kind == PREV
    assert parse_command("j5") == Command(JUMP, value=5)
    assert parse_command("q").kind == QUIT
    assert parse_command("?").kind == HELP


def test_parse_reject_code_commands():
    cmd = parse_command("x bad_question")
    assert cmd.kind == REJECT_CMD and cmd.field == "bad_question"
    assert parse_command("x").kind == REJECT_CMD and parse_command("x").field == ""


# --- format_card --------------------------------------------------------------------------


def test_format_card_hides_crosscheck_by_default():
    row = {col: "" for col in WORKSHEET_COLS}
    row.update({"item_id": "a", "question": "q", "context": "ctx>>>x<<<", "cc_supported": "false"})
    assert "crosscheck" not in format_card(row, 1, 1, 0)
    assert "crosscheck" in format_card(row, 1, 1, 0, show_crosscheck=True)


def test_format_card_omits_planted_for_real_items():
    row = {col: "" for col in WORKSHEET_COLS}
    row.update({"item_id": "a", "synthetic": "false"})
    assert "chk_planted" not in format_card(row, 1, 1, 0)
    row["synthetic"] = "true"
    assert "chk_planted" in format_card(row, 1, 1, 0)


# --- the interactive loop (injected I/O) --------------------------------------------------


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


# --- coded rejection reasons in the session -------------------------------------------------


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


# --- accept-with-edit re-grounding in the session -------------------------------------------


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


# --- session throughput stats ---------------------------------------------------------------


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
