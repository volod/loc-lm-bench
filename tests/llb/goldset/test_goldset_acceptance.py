"""Tests for goldset acceptance."""

import json
import pytest
from llb.bench.common import verified_data_config
from llb.goldset.schema import load_goldset
from llb.goldset.verify_acceptance import accepted_ids, emit_accepted_ledger
from llb.goldset.verify_acceptance_report import acceptance_report
from llb.goldset.verify_ref_format import format_verification_status
from llb.goldset.verify_refcheck import check_verification_ref
from llb.prep.verified_ledger import apply_verified_ledger, load_verified_ledger
from tests.llb.goldset._verify_helpers import (
    DOC,
    _bundle,
    _item,
    _ws_row,
)


def test_acceptance_pass_within_tolerance():
    rows = [_ws_row(f"a{i}", "accept") for i in range(19)] + [_ws_row("r0", "reject")]
    report = acceptance_report(rows, tolerance=0.05)
    assert report["decided"] == 20 and report["rejected"] == 1
    assert report["reject_rate"] == 0.05 and report["passed"] is True


def test_acceptance_fail_over_tolerance():
    rows = [_ws_row("a0", "accept"), _ws_row("r0", "reject"), _ws_row("r1", "reject")]
    report = acceptance_report(rows, tolerance=0.05)
    assert report["passed"] is False


def test_acceptance_flags_undecided_failures():
    rows = [_ws_row("a0", "accept"), _ws_row("u0", "", chk_grounded="fail")]
    report = acceptance_report(rows)
    assert report["undecided"] == 1 and report["undecided_with_failures"] == 1


def test_check_verification_ref_accepts_decided_worksheet(tmp_path):
    path = tmp_path / "verify_sample.csv"
    path.write_text("item_id,stratum,decision\nok,s,accept\n", encoding="utf-8")
    status = check_verification_ref(path)
    assert status.valid is True and status.kind == "worksheet"


def test_check_verification_ref_rejects_undecided_worksheet(tmp_path):
    path = tmp_path / "verify_sample.csv"
    path.write_text("item_id,stratum,decision\nok,s,\n", encoding="utf-8")
    status = check_verification_ref(path)
    assert status.valid is False and "undecided" in status.reason
    assert status.stats["undecided"] == 1
    message = format_verification_status(status)
    assert "stats:" in message
    assert "undecided: 1" in message
    assert "make verify-review VERIFY_WS=" in message
    assert "--data-verified --verification-ref" in message


def test_verified_data_config_rejects_invalid_ref_with_operator_diagnostics(tmp_path):
    path = tmp_path / "verify_sample.csv"
    path.write_text("item_id,stratum,decision\nok,s,\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        verified_data_config(data_verified=True, verification_ref=str(path))

    message = str(excinfo.value)
    assert "verification reference cannot be used with --data-verified" in message
    assert "undecided: 1" in message
    assert "make verify-review VERIFY_WS=" in message


def test_check_verification_ref_accepts_sample_manifest(tmp_path):
    path = tmp_path / "verify_sample.csv"
    path.write_text("item_id,stratum,decision\nok,s,accept\n", encoding="utf-8")
    manifest = tmp_path / "sample_manifest.json"
    manifest.write_text(json.dumps({"worksheet": str(path)}), encoding="utf-8")
    status = check_verification_ref(manifest)
    assert status.valid is True and status.kind == "sample_manifest"


def test_per_stratum_failure_is_isolated():
    rows = [_ws_row(f"x{i}", "accept", stratum="clean") for i in range(10)]
    rows += [_ws_row("y0", "reject", stratum="dirty"), _ws_row("y1", "accept", stratum="dirty")]
    report = acceptance_report(rows, tolerance=0.05)
    assert report["per_stratum"]["clean"]["passed"] == 1.0
    assert report["per_stratum"]["dirty"]["passed"] == 0.0  # 50% reject hides at the overall level


def test_emit_accepted_ledger_round_trips_through_the_ledger(tmp_path):
    bundle = _bundle(tmp_path, [_item("keep"), _item("drop")])
    out_dir = tmp_path / "accepted"
    n = emit_accepted_ledger(bundle, ["keep"], out_dir)
    assert n == 1
    accepted = load_goldset(out_dir / "goldset.jsonl")
    assert accepted[0].id == "keep" and accepted[0].verified is True
    assert (out_dir / "corpus" / DOC).is_file()  # grounding doc copied -> self-contained
    # The ingester adopts the accepted id by REPLACEMENT (not a boolean flip on the draft).
    ledger = load_verified_ledger([out_dir / "goldset.jsonl"])
    drafts = [_item("keep"), _item("drop")]
    merged, docs, n_verified = apply_verified_ledger(drafts, ledger)
    assert n_verified == 1
    assert next(i for i in merged if i.id == "keep").verified is True
    assert next(i for i in merged if i.id == "drop").verified is False


def test_check_verification_ref_accepts_accepted_ledger(tmp_path):
    bundle = _bundle(tmp_path / "draft", [_item("keep")])
    out_dir = tmp_path / "accepted"
    emit_accepted_ledger(bundle, ["keep"], out_dir)
    status = check_verification_ref(out_dir)
    assert status.valid is True and status.kind == "accepted_ledger"


def test_accepted_ids_only_accept_decisions():
    rows = [_ws_row("a", "accept"), _ws_row("r", "reject"), _ws_row("u", "")]
    assert accepted_ids(rows) == ["a"]
