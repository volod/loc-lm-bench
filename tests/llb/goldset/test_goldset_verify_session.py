"""Tests for the interactive verification session (`llb.goldset.verify_session` + `verify_card`).

The session loop is driven by an INJECTED input iterator + output sink, so no terminal / model /
endpoint / GPU is needed -- it operates only on the CSV. Card rendering and command parsing are
checked directly. Shared factories live in `_verify_helpers.py`; the pure verify pieces are in
`test_goldset_verify.py`.
"""

import json

from llb.goldset.verify_base import WORKSHEET_COLS, load_worksheet
from llb.goldset.verify_sampling.worksheet import build_sample_worksheet
from llb.goldset.verify_card import format_card
from llb.goldset.verify_commands import (
    ACCEPT_CMD,
    CHECK,
    HELP,
    JUMP,
    NEXT,
    PREV,
    QUIT,
    REJECT_CMD,
    Command,
    parse_command,
)
from llb.goldset.verify_session.loop import run_session

from tests.llb.goldset._verify_helpers import (
    _chain,
    _chain_bundle,
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


# --- coded rejection reasons in the session -------------------------------------------------


# --- accept-with-edit re-grounding in the session -------------------------------------------


# --- session throughput stats ---------------------------------------------------------------
