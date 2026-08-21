"""Measured claim-tier precision: the calibration gate, the clustered bound, and the rendering.

The quantity under test is the share of the RETURNED candidate list that survives adjudication.
Two properties are what make it publishable and both are asserted here: the bound is the same
two-way clustered estimator the independent-null research established (asserted by running the
research lane's own curve over the same rows), and the block is suppressed with a stated reason
whenever the adjudicator that produced the verdicts has not cleared its calibration bound.
"""

import json
from types import SimpleNamespace

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.claim.calibration import (
    MIN_ADJUDICATOR_ACCURACY_LCB,
    calibrate_adjudicator,
    load_calibration_probe,
)
from llb.conflicts.claim.precision import precision_block, unparsed_allowance
from llb.conflicts.constants import TIER_CLAIM
from llb.conflicts.null_research.statistics.precision import CandidateRow, precision_curve
from llb.conflicts.report.render import render_report, write_audit
from tests.llb.conflicts.conflict_helpers import (
    FAKE_COS_THRESHOLD,
    FIXTURE_CORPUS,
    adjudicated_rows,
    calibrated_stub,
    fake_store_view,
    probe_aware,
)
from tests.llb.conflicts.test_audit import scripted

SEED = 20260812


# --- the frozen probe -------------------------------------------------------------------------


def test_probe_resolves_to_fixture_passages_and_is_balanced():
    probe = load_calibration_probe()
    assert len(probe.pairs) == 24
    assert sum(pair.actionable for pair in probe.pairs) == 12, "half the probe must be actionable"
    for pair in probe.pairs:
        assert pair.left_text.startswith("#") and pair.right_text.startswith("#")
        assert len(pair.left_text.split()) > 5 and len(pair.right_text.split()) > 5
    assert len({pair.pair_id for pair in probe.pairs}) == len(probe.pairs)


def test_probe_refuses_a_heading_the_fixture_does_not_have(tmp_path):
    payload = json.loads((FIXTURE_CORPUS.parent / "adjudicator_probe.json").read_text("utf-8"))
    payload["pairs"][0]["a"]["heading"] = "## Розділ 9. Немає такого"
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="has no heading"):
        load_calibration_probe(path)


def test_a_frozen_label_adjudicator_calibrates():
    calibration = calibrate_adjudicator(load_calibration_probe(), probe_aware(scripted))
    assert calibration["accuracy"] == 1.0
    assert calibration["calibrated"] and calibration["unparsed_pairs"] == 0
    assert calibration["recall_on_actionable"] == 1.0
    assert calibration["specificity_on_complementary"] == 1.0


def test_an_adjudicator_that_calls_everything_actionable_does_not_calibrate():
    calibration = calibrate_adjudicator(
        load_calibration_probe(), probe_aware(scripted, correct=False)
    )
    assert calibration["accuracy"] == 0.5
    assert calibration["specificity_on_complementary"] == 0.0
    assert not calibration["calibrated"]
    assert calibration["accuracy_wilson_95"][0] < MIN_ADJUDICATOR_ACCURACY_LCB


# --- the bound --------------------------------------------------------------------------------


def test_the_printed_bound_equals_the_research_harness_bound_on_the_same_rows():
    """The acceptance gate: one estimator, so the audit cannot quote a looser bound than research."""
    flags = [True, True, False, True, False, True, True, False, True, True, False, True]
    left_keys = [f"L{index // 2}" for index in range(len(flags))]
    right_keys = [f"R{index % 5}" for index in range(len(flags))]
    rows = adjudicated_rows(flags, left_keys=left_keys, right_keys=right_keys)

    block = precision_block(rows, calibrated_stub(), seed=SEED)

    corpus = SimpleNamespace(
        chunks=[{"chunk_id": key, "doc_id": "d"} for key in left_keys + right_keys]
    )
    research_rows = [
        CandidateRow(left=index, right=len(flags) + index, score=row.score)
        for index, row in enumerate(rows)
    ]
    verdicts = [{"actionable": flag, "parsed": True} for flag in flags]
    assert block["precision_curve"] == precision_curve(corpus, research_rows, verdicts, seed=SEED)


def test_the_returned_budget_point_is_the_whole_adjudicated_list():
    rows = adjudicated_rows([True] * 8 + [False] * 4)
    point = precision_block(rows, calibrated_stub(), seed=SEED)["returned_budget"]
    assert (point["budget"], point["actionable_rows"]) == (12, 8)
    assert point["precision"] == pytest.approx(8 / 12)
    assert 0.0 <= point["two_way_clustered_lcb"] <= point["precision"]


def test_repeating_the_same_chunks_buys_pair_row_confidence_but_not_clustered_confidence():
    """Restating six units as twenty-four rows adds no evidence, and only Wilson is fooled."""
    flags = [True] * 5 + [False]
    units = precision_block(adjudicated_rows(flags), calibrated_stub(), seed=SEED)[
        "returned_budget"
    ]
    repeated = precision_block(
        adjudicated_rows(
            flags * 4,
            left_keys=[f"L{index % 6}" for index in range(24)],
            right_keys=[f"R{index % 6}" for index in range(24)],
        ),
        calibrated_stub(),
        seed=SEED,
    )["returned_budget"]
    assert units["precision"] == repeated["precision"]
    assert repeated["wilson_95"][0] > units["wilson_95"][0] + 0.15, "pair rows look 4x stronger"
    assert repeated["two_way_clustered_lcb"] == pytest.approx(
        units["two_way_clustered_lcb"], abs=0.1
    ), "the clustered bound rests on the six chunks, not on the twenty-four rows"


def test_an_unparsed_verdict_counts_against_precision_rather_than_vanishing():
    rows = adjudicated_rows([True] * 19 + [False], parsed=[True] * 19 + [False])
    block = precision_block(rows, calibrated_stub(), seed=SEED)
    assert block["unparsed_rows"] == 1
    assert block["returned_budget"]["actionable_rows"] == 19
    assert block["returned_budget"]["budget"] == 20


def test_one_unparsable_verdict_does_not_erase_a_twelve_row_measurement():
    """The bias is downward, so the figure stays a lower bound rather than disappearing."""
    rows = adjudicated_rows([True] * 11 + [False], parsed=[True] * 11 + [False])
    block = precision_block(rows, calibrated_stub(), seed=SEED)
    assert unparsed_allowance(12) == 1
    assert block["reported"], block.get("reason")
    assert block["returned_budget"]["actionable_rows"] == 11


# --- the suppression gate ---------------------------------------------------------------------


def test_precision_is_suppressed_without_calibration():
    block = precision_block(adjudicated_rows([True] * 6), None, seed=SEED)
    assert not block["reported"] and "returned_budget" not in block
    assert "calibration was not run" in block["reason"]


def test_precision_is_suppressed_when_the_adjudicator_misses_its_bound():
    calibration = {**calibrated_stub(), "calibrated": False, "accuracy_wilson_95": [0.41, 0.72]}
    block = precision_block(adjudicated_rows([True] * 6), calibration, seed=SEED)
    assert not block["reported"] and "returned_budget" not in block
    assert "missed its calibration bound" in block["reason"] and "0.41" in block["reason"]


def test_precision_is_suppressed_when_too_many_verdicts_are_unparsable():
    rows = adjudicated_rows([True] * 10, parsed=[True] * 8 + [False, False])
    block = precision_block(rows, calibrated_stub(), seed=SEED)
    assert not block["reported"]
    assert "unparsable verdict" in block["reason"]


def test_precision_is_suppressed_when_no_row_was_adjudicated():
    block = precision_block([], calibrated_stub(), seed=SEED)
    assert not block["reported"] and "no candidate rows" in block["reason"]


# --- the audit end to end ---------------------------------------------------------------------


def _audit(**kwargs):
    return run_audit(
        FIXTURE_CORPUS,
        AuditParams(effort=TIER_CLAIM, cos_threshold=FAKE_COS_THRESHOLD, **kwargs),
        store=fake_store_view(),
        complete=probe_aware(scripted),
    )


def test_a_claim_audit_carries_a_calibrated_precision_block_into_both_artifacts(tmp_path):
    result = _audit()
    block = result.claim_precision
    assert block["reported"], block.get("reason")
    assert block["adjudicator_calibration"]["calibrated"]
    assert block["returned_budget"]["budget"] == block["adjudicated_rows"]
    assert len(block["rows"]) == block["adjudicated_rows"]

    paths = write_audit(tmp_path, result)
    summary = json.loads(paths["summary"].read_text("utf-8"))
    assert summary["claim_precision"]["returned_budget"] == block["returned_budget"]
    report = paths["report"].read_text("utf-8")
    assert "## Claim-tier precision" in report
    assert "two-way clustered 95% lower bound" in report
    assert "frozen probe pairs agree" in report


def test_an_uncalibrated_audit_states_why_precision_is_missing():
    result = run_audit(
        FIXTURE_CORPUS,
        AuditParams(effort=TIER_CLAIM, cos_threshold=FAKE_COS_THRESHOLD),
        store=fake_store_view(),
        complete=probe_aware(scripted, correct=False),
    )
    assert not result.claim_precision["reported"]
    report = render_report(result)
    section = report.split("## Claim-tier precision")[1].split("\n## ")[0]
    assert "Not reported: the adjudicator missed its calibration bound" in section
    assert "two-way clustered 95% lower bound" not in section, "no figure without the calibration"
    assert "frozen probe pairs agree" in section, "the failed calibration is still shown"


def test_skipping_calibration_costs_no_model_calls_and_suppresses_the_block():
    calls: list[str] = []

    def counting(prompt: str) -> str:
        calls.append(prompt)
        return scripted(prompt)

    result = run_audit(
        FIXTURE_CORPUS,
        AuditParams(
            effort=TIER_CLAIM, cos_threshold=FAKE_COS_THRESHOLD, calibrate_adjudicator=False
        ),
        store=fake_store_view(),
        complete=counting,
    )
    claim = next(stat for stat in result.tiers if stat.tier == TIER_CLAIM)
    assert len(calls) == claim.extra["model_calls"], "no probe calls when calibration is skipped"
    assert not result.claim_precision["reported"]
    assert result.claim_precision["adjudicator_calibration"] is None
