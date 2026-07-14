"""Tests for verify adjudication flow."""

import json
import pytest
from llb.goldset.schema import load_goldset
from llb.goldset.verify_acceptance import run_accept
from llb.goldset.verify_acceptance_report import (
    acceptance_report,
    confidence_weighted_reject_rate,
)
from llb.goldset.verify_base import (
    POLICY_GLOBAL,
    POLICY_PER_STRATUM,
    POLICY_WEIGHTED,
    load_worksheet,
    write_worksheet_rows,
)
from llb.goldset.verify_multi.adjudication import build_adjudication_worksheet, run_adjudicate
from llb.goldset.verify_multi.common import (
    ADJUDICATION_FILENAME,
    ADJUDICATOR_ID,
    AGREEMENT_FILENAME,
    PRIOR_DECISIONS_COL,
    reviewer_worksheet_path,
)
from llb.goldset.verify_multi.consensus import ConsensusBuilder, resolve_multi_reviewer_rows
from llb.goldset.verify_multi.sampling import build_multi_reviewer_worksheets
from llb.goldset.verify_card import format_card
from llb.goldset.verify_session.loop import run_session
from test_verify_adjudication import (
    _bundle,
    _decide,
    _item,
    _stratified_rows,
    _two_reviewers,
    _ws_row,
)


def test_adjudication_draws_exactly_disagreements_and_carries_priors(tmp_path):
    by_reviewer = _two_reviewers(["accept", "reject", "accept"], ["accept", "accept", "accept"])
    by_reviewer["r1"][1]["reject_code"] = "bad_question"
    base = tmp_path / "verify_sample.csv"
    path, n = build_adjudication_worksheet(base, by_reviewer, ["i1"])
    assert path.name == ADJUDICATION_FILENAME and n == 1
    rows, fields = load_worksheet(path)
    assert PRIOR_DECISIONS_COL in fields
    assert [r["item_id"] for r in rows] == ["i1"]
    assert rows[0][PRIOR_DECISIONS_COL] == "r1=reject:bad_question;r2=accept"
    assert rows[0]["reviewer_id"] == ADJUDICATOR_ID
    assert rows[0]["decision"] == ""  # fresh, independent decision


def test_adjudication_rebuild_preserves_adjudicator_decisions(tmp_path):
    by_reviewer = _two_reviewers(["reject", "reject"], ["accept", "accept"])
    base = tmp_path / "verify_sample.csv"
    path, _ = build_adjudication_worksheet(base, by_reviewer, ["i0", "i1"])
    rows, fields = load_worksheet(path)
    rows[0]["decision"] = "accept"
    write_worksheet_rows(path, rows, fields)
    path, n = build_adjudication_worksheet(base, by_reviewer, ["i0", "i1"])
    assert n == 2
    rows, _ = load_worksheet(path)
    assert {r["item_id"]: r["decision"] for r in rows} == {"i0": "accept", "i1": ""}


def test_consensus_unanimous_stands_disagreement_blocks_adjudication_overrides():
    by_reviewer = _two_reviewers(
        ["accept", "reject", "accept", ""], ["accept", "accept", "accept", "accept"]
    )
    # i1 disagreement adjudicated to reject; i2 unanimous accept; i3 half-undecided.
    adjudication = [_ws_row("i1", "reject", reviewer_id=ADJUDICATOR_ID)]
    merged = ConsensusBuilder(by_reviewer, adjudication).build()
    decisions = {r["item_id"]: r["decision"] for r in merged}
    assert decisions == {"i0": "accept", "i1": "reject", "i2": "accept", "i3": ""}


def test_consensus_unadjudicated_disagreement_stays_undecided():
    by_reviewer = _two_reviewers(["reject"], ["accept"])
    merged = ConsensusBuilder(by_reviewer).build()
    assert merged[0]["decision"] == ""
    report = acceptance_report(merged)
    assert report["undecided"] == 1 and report["passed"] is False


def test_policy_global_passes_where_per_stratum_fails():
    rows = _stratified_rows()  # overall 1/32 = 0.031 <= 0.05, dirty stratum 0.5 > 0.05
    assert acceptance_report(rows, 0.05, policy=POLICY_GLOBAL)["passed"] is True
    assert acceptance_report(rows, 0.05, policy=POLICY_PER_STRATUM)["passed"] is False


def test_policy_per_stratum_honors_overrides():
    rows = _stratified_rows()
    report = acceptance_report(
        rows, 0.05, policy=POLICY_PER_STRATUM, stratum_tolerances={"dirty": 0.6}
    )
    assert report["passed"] is True
    assert report["per_stratum"]["dirty"]["tolerance"] == 0.6


def test_policy_weighted_hand_computed():
    # accept with confident signals (weight 2.0), reject the signals flagged (weight 1.0):
    # weighted rate = 1/3; a confident reject flips the weights: rate = 2/3.
    lenient = [
        _ws_row("a", "accept", cc_grounded="true"),
        _ws_row("b", "reject", cc_grounded="false"),
    ]
    assert confidence_weighted_reject_rate(lenient) == pytest.approx(1.0 / 3.0)
    harsh = [
        _ws_row("a", "reject", cc_grounded="true"),
        _ws_row("b", "accept", cc_grounded="false"),
    ]
    assert confidence_weighted_reject_rate(harsh) == pytest.approx(2.0 / 3.0)
    report = acceptance_report(lenient, 0.4, policy=POLICY_WEIGHTED)
    assert report["passed"] is True and report["reject_rate"] == 0.5  # global would fail
    assert acceptance_report(harsh, 0.4, policy=POLICY_WEIGHTED)["passed"] is False


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="policy"):
        acceptance_report([_ws_row("a", "accept")], policy="majority")


def test_multi_reviewer_accept_scores_consensus_and_emits_ledger(tmp_path, caplog):
    bundle = _bundle(tmp_path, [_item("a"), _item("b"), _item("c")])
    base = bundle / "verify_sample.csv"
    build_multi_reviewer_worksheets(bundle, base, n=3, annotators=2, seed=1)
    _decide(reviewer_worksheet_path(base, 1), {"a": "accept", "b": "accept", "c": "accept"})
    _decide(reviewer_worksheet_path(base, 2), {"a": "accept", "b": "reject", "c": "accept"})

    assert run_adjudicate(bundle) == 0
    agreement = json.loads((bundle / AGREEMENT_FILENAME).read_text(encoding="utf-8"))
    assert agreement["disagreements"] == ["b"]
    adj = bundle / ADJUDICATION_FILENAME
    _decide(adj, {"b": "accept"})

    # verify-accept passes the BASE worksheet path; the manifest routes it to the consensus.
    assert run_accept(base, bundle, None, tolerance=0.05) == 0
    accepted = load_goldset(bundle / "accepted" / "goldset.jsonl")
    assert sorted(item.id for item in accepted) == ["a", "b", "c"]
    assert all(item.verified for item in accepted)


def test_resolve_multi_reviewer_rows_is_none_for_single_worksheet(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    from llb.goldset.verify_sampling.worksheet import build_sample_worksheet

    ws = bundle / "verify_sample.csv"
    build_sample_worksheet(bundle, ws, n=1)
    assert resolve_multi_reviewer_rows(ws) is None  # single-reviewer flow unchanged


def test_card_shows_reviewer_and_prior_decisions():
    row = _ws_row("a", reviewer_id="adjudicator")
    row[PRIOR_DECISIONS_COL] = "r1=reject:bad_question;r2=accept"
    card = format_card(row, 1, 1, 0)
    assert "reviewer: adjudicator" in card
    assert "r1=reject:bad_question;r2=accept" in card
    plain = format_card(_ws_row("a"), 1, 1, 0)
    assert "reviewer:" not in plain and "prior_decisions" not in plain


def test_session_reviews_adjudication_worksheet(tmp_path):
    by_reviewer = _two_reviewers(["reject"], ["accept"])
    base = tmp_path / "verify_sample.csv"
    path, _ = build_adjudication_worksheet(base, by_reviewer, ["i0"])
    out: list[str] = []
    decided = run_session(path, inputs=iter(["y", "q"]), output=out.append)
    assert decided == 1
    rows, _ = load_worksheet(path)
    assert rows[0]["decision"] == "accept"
    assert rows[0][PRIOR_DECISIONS_COL] == "r1=reject;r2=accept"  # read-only column survived
