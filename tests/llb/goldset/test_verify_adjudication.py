"""Tests for the multi-annotator verification gate (`llb.goldset.verify_multi`) and the
configurable acceptance arithmetic in `llb.goldset.verify`.

Everything runs on synthetic reviewed fixtures: agreement statistics are checked against
hand-computed kappa values, the adjudication draw against constructed disagreements, and each
acceptance policy against a worksheet whose pass/fail flips with the policy.
"""

import json

import pytest

from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset, load_goldset
from llb.goldset.verify_acceptance import (
    acceptance_report,
    confidence_weighted_reject_rate,
    run_accept,
)
from llb.goldset.verify_base import (
    POLICY_GLOBAL,
    POLICY_PER_STRATUM,
    POLICY_WEIGHTED,
    WORKSHEET_COLS,
    load_worksheet,
    write_worksheet_rows,
)
from llb.goldset.verify_multi.adjudication import build_adjudication_worksheet, run_adjudicate
from llb.goldset.verify_multi.agreement import (
    agreement_report,
    cohen_kappa,
    fleiss_kappa,
)
from llb.goldset.verify_multi.common import (
    ADJUDICATION_FILENAME,
    ADJUDICATOR_ID,
    AGREEMENT_FILENAME,
    PRIOR_DECISIONS_COL,
    load_reviewer_worksheets,
    reviewer_worksheet_path,
)
from llb.goldset.verify_multi.consensus import consensus_rows, resolve_multi_reviewer_rows
from llb.goldset.verify_multi.sampling import build_multi_reviewer_worksheets
from llb.goldset.verify_card import format_card
from llb.goldset.verify_session.loop import run_session

DOC = "squad/doc1.txt"
TEXT = "Леся Українка народилася 1871 року в Новограді-Волинському. Вона була поетесою."


def _item(item_id, *, answer="1871"):
    start = TEXT.find(answer)
    return GoldItem(
        id=item_id,
        question=f"Коли подія {item_id}?",
        reference_answer=answer,
        source_doc_id=DOC,
        source_spans=[
            SourceSpan(doc_id=DOC, char_start=start, char_end=start + len(answer), text=answer)
        ],
        provenance="frontier-drafted",
        split="calibration",
    )


def _bundle(tmp_path, items):
    dump_goldset(items, tmp_path / "goldset.jsonl")
    doc = tmp_path / "corpus" / DOC
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(TEXT + "\n", encoding="utf-8")
    return tmp_path


def _ws_row(item_id, decision="", stratum="s", **over):
    row = {col: "" for col in WORKSHEET_COLS}
    row.update({"item_id": item_id, "stratum": stratum, "decision": decision})
    row.update(over)
    return row


# --- agreement math against hand-computed fixtures -------------------------------------------


def test_cohen_kappa_hand_computed():
    # po = 3/4; pe = 0.75*0.5 + 0.25*0.5 = 0.5 -> kappa = (0.75 - 0.5) / 0.5 = 0.5
    a = ["accept", "accept", "reject", "accept"]
    b = ["accept", "reject", "reject", "accept"]
    assert cohen_kappa(a, b) == pytest.approx(0.5)


def test_cohen_kappa_degenerate_cases():
    assert cohen_kappa(["accept", "accept"], ["accept", "accept"]) == 1.0
    assert cohen_kappa(["accept", "accept"], ["reject", "reject"]) == 0.0
    with pytest.raises(ValueError):
        cohen_kappa(["accept"], ["accept", "reject"])


def test_fleiss_kappa_hand_computed():
    # 3 raters x 4 items over (accept, reject):
    # P_i = [1, 1/3, 1/3, 1]; P_bar = 2/3; p = (0.75, 0.25); Pe = 0.625
    # kappa = (2/3 - 0.625) / 0.375 = 1/9
    counts = [[3, 0], [2, 1], [1, 2], [3, 0]]
    assert fleiss_kappa(counts) == pytest.approx(1.0 / 9.0)


def test_fleiss_kappa_guards():
    assert fleiss_kappa([[2, 0], [2, 0]]) == 1.0  # constant raters, perfect agreement
    with pytest.raises(ValueError):
        fleiss_kappa([[2, 0], [3, 0]])  # unequal rater counts
    with pytest.raises(ValueError):
        fleiss_kappa([[1, 0]])  # one rater


# --- multi-reviewer sampling ------------------------------------------------------------------


def test_multi_sample_writes_identical_per_reviewer_worksheets(tmp_path):
    bundle = _bundle(tmp_path, [_item("a"), _item("b"), _item("c")])
    base = bundle / "verify_sample.csv"
    paths = build_multi_reviewer_worksheets(bundle, base, n=2, annotators=3, seed=1)
    assert [p.name for p in paths] == [
        "verify_sample.r1.csv",
        "verify_sample.r2.csv",
        "verify_sample.r3.csv",
    ]
    by_reviewer = load_reviewer_worksheets(paths)
    ids = {rid: [r["item_id"] for r in rows] for rid, rows in by_reviewer.items()}
    assert ids["r1"] == ids["r2"] == ids["r3"]  # SAME sample for every reviewer
    assert all(row["reviewer_id"] == rid for rid, rows in by_reviewer.items() for row in rows)
    manifest = json.loads((bundle / "sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest["annotators"] == 3
    assert len(manifest["worksheets"]) == 3
    # No single-`worksheet` key: one reviewer's sheet alone can never stamp --data-verified.
    assert "worksheet" not in manifest


def test_multi_sample_requires_two_annotators(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    with pytest.raises(ValueError, match="annotators"):
        build_multi_reviewer_worksheets(bundle, bundle / "ws.csv", n=1, annotators=1)


# --- agreement report -------------------------------------------------------------------------


def _two_reviewers(decisions_a, decisions_b, **row_kw):
    ids = [f"i{k}" for k in range(len(decisions_a))]
    return {
        "r1": [_ws_row(i, d, reviewer_id="r1", **row_kw) for i, d in zip(ids, decisions_a)],
        "r2": [_ws_row(i, d, reviewer_id="r2", **row_kw) for i, d in zip(ids, decisions_b)],
    }


def test_agreement_report_two_reviewers_matches_cohen():
    by_reviewer = _two_reviewers(
        ["accept", "accept", "reject", "accept"], ["accept", "reject", "reject", "accept"]
    )
    report = agreement_report(by_reviewer)
    assert report["kappa_method"] == "cohen"
    assert report["kappa"] == pytest.approx(0.5)
    assert report["jointly_decided"] == 4
    assert report["observed_agreement"] == pytest.approx(0.75)
    assert report["disagreements"] == ["i1"]
    assert report["per_reviewer"]["r2"] == {"decided": 4, "accepted": 2, "rejected": 2}


def test_agreement_report_three_reviewers_uses_fleiss():
    ids = ["i0", "i1", "i2", "i3"]
    decisions = {
        "r1": ["accept", "accept", "reject", "accept"],
        "r2": ["accept", "accept", "reject", "accept"],
        "r3": ["accept", "reject", "accept", "accept"],
    }
    by_reviewer = {
        rid: [_ws_row(i, d, reviewer_id=rid) for i, d in zip(ids, decisions[rid])]
        for rid in decisions
    }
    report = agreement_report(by_reviewer)
    assert report["kappa_method"] == "fleiss"
    # counts per item over (accept, reject): [3,0], [2,1], [1,2], [3,0] -> kappa = 1/9
    assert report["kappa"] == pytest.approx(1.0 / 9.0)
    assert report["disagreements"] == ["i1", "i2"]


def test_agreement_report_undecided_rows_do_not_count():
    by_reviewer = _two_reviewers(["accept", ""], ["accept", "reject"])
    report = agreement_report(by_reviewer)
    assert report["jointly_decided"] == 1  # i1 is not decided by r1 -> not a disagreement
    assert report["disagreements"] == []
    assert report["kappa"] is None  # fewer than 2 jointly decided rows


def test_agreement_flags_unanimous_accept_with_differing_edits():
    by_reviewer = _two_reviewers(["accept", "accept"], ["accept", "accept"])
    by_reviewer["r1"][1]["edited_answer"] = "1871 року"
    report = agreement_report(by_reviewer)
    assert report["disagreements"] == ["i1"]  # the edit changes what the ledger would certify


# --- adjudication worksheet -------------------------------------------------------------------


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


# --- consensus --------------------------------------------------------------------------------


def test_consensus_unanimous_stands_disagreement_blocks_adjudication_overrides():
    by_reviewer = _two_reviewers(
        ["accept", "reject", "accept", ""], ["accept", "accept", "accept", "accept"]
    )
    # i1 disagreement adjudicated to reject; i2 unanimous accept; i3 half-undecided.
    adjudication = [_ws_row("i1", "reject", reviewer_id=ADJUDICATOR_ID)]
    merged = consensus_rows(by_reviewer, adjudication)
    decisions = {r["item_id"]: r["decision"] for r in merged}
    assert decisions == {"i0": "accept", "i1": "reject", "i2": "accept", "i3": ""}


def test_consensus_unadjudicated_disagreement_stays_undecided():
    by_reviewer = _two_reviewers(["reject"], ["accept"])
    merged = consensus_rows(by_reviewer)
    assert merged[0]["decision"] == ""
    report = acceptance_report(merged)
    assert report["undecided"] == 1 and report["passed"] is False


# --- acceptance policies ----------------------------------------------------------------------


def _stratified_rows():
    rows = [_ws_row(f"c{i}", "accept", stratum="clean") for i in range(30)]
    rows += [_ws_row("d0", "reject", stratum="dirty"), _ws_row("d1", "accept", stratum="dirty")]
    return rows


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


# --- end-to-end: multi-reviewer accept through the ledger --------------------------------------


def _decide(path, decisions):
    rows, fields = load_worksheet(path)
    for row in rows:
        if row["item_id"] in decisions:
            row["decision"] = decisions[row["item_id"]]
    write_worksheet_rows(path, rows, fields)


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


# --- session integration ----------------------------------------------------------------------


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
