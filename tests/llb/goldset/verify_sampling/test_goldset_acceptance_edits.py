"""Tests for goldset acceptance edits."""

import json
import pytest
from llb.goldset.schema import load_goldset
from llb.goldset.verify_acceptance import (
    emit_accepted_ledger,
    run_accept,
)
from llb.goldset.verify_acceptance_report import (
    ground_answer,
    infer_reject_code,
    rejection_reasons_summary,
)
from llb.goldset.verify_base import load_worksheet, write_worksheet_rows
from llb.goldset.verify_sampling.worksheet import (
    build_sample_worksheet,
)
from tests.llb.goldset._verify_helpers import (
    TEXT,
    _bundle,
    _item,
    _ws_row,
)


def test_infer_reject_code_prefers_first_failed_check():
    assert infer_reject_code(_ws_row("a", chk_reference="fail")) == "wrong_reference"
    assert (
        infer_reject_code(_ws_row("a", chk_grounded="fail", chk_reference="fail")) == "ungrounded"
    )
    assert infer_reject_code(_ws_row("a")) == "other"


def test_run_accept_exports_rejection_reasons(tmp_path):
    bundle = _bundle(tmp_path, [_item("a"), _item("b")])
    out = tmp_path / "ws.csv"
    build_sample_worksheet(bundle, out, n=2, seed=1)
    rows, fields = load_worksheet(out)
    rows[0]["decision"] = "accept"
    rows[1]["decision"] = "reject"
    rows[1]["reject_code"] = "bad_question"
    rows[1]["human_note"] = "тривiальне"
    write_worksheet_rows(out, rows, fields)
    assert run_accept(out, bundle, None, tolerance=0.6) == 0
    reasons = json.loads(
        (bundle / "accepted" / "rejection_reasons.json").read_text(encoding="utf-8")
    )
    assert reasons["rejected"] == 1
    cell = reasons["by_code"]["bad_question"]
    assert cell["count"] == 1 and cell["items"][0]["note"] == "тривiальне"


def test_rejection_reasons_summary_infers_missing_codes():
    rows = [_ws_row("a", "reject", chk_grounded="fail"), _ws_row("b", "accept")]
    summary = rejection_reasons_summary(rows)
    assert summary["rejected"] == 1 and "ungrounded" in summary["by_code"]


def test_ground_answer_prefers_occurrence_nearest_hint():
    text = "рік 1871 та ще раз 1871 у кінці"
    late = text.rfind("1871")
    assert ground_answer(text, "1871", hint_start=late) == (late, late + 4)
    assert ground_answer(text, "відсутнє") is None


def test_emit_accepted_ledger_applies_regrounded_edit(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    out_dir = tmp_path / "accepted"
    n = emit_accepted_ledger(bundle, ["a"], out_dir, edits={"a": "Новограді-Волинському"})
    assert n == 1
    item = load_goldset(out_dir / "goldset.jsonl")[0]
    assert item.verified is True
    assert item.reference_answer == "Новограді-Волинському"
    assert item.source_spans[0].text == "Новограді-Волинському"
    start = item.source_spans[0].char_start
    assert TEXT[start : start + len("Новограді-Волинському")] == "Новограді-Волинському"


def test_emit_accepted_ledger_blocks_ungrounded_edit(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    with pytest.raises(ValueError, match="verbatim"):
        emit_accepted_ledger(
            bundle, ["a"], tmp_path / "accepted", edits={"a": "вигадана вiдповiдь"}
        )
