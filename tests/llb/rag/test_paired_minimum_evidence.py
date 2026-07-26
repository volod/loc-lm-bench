"""paired-reading-minimum-evidence-gate -- a reading no item count could support is not one.

A percentile bootstrap of a handful of DISCORDANT items happily prints `+1.000 [+1.000, +1.000]`
beside its own exact sign-test p of 0.5: resampling two differing items can only ever draw those
two. The gate refuses to call that a separation, and the bound it refuses at is derived from the
reporting confidence -- the fewest differing items the exact two-sided sign test could reach that
level with -- so nothing here is fitted.

Covered: the derived bound and its behaviour at the 5 / 6 / 7 boundary, that only the CLAIM is
gated (never the null reading, never the numbers), and the verdict guard in every lane that cuts a
`lo > 0` reading.

Pure: value vectors and dict rows, so the whole vertical runs in the lightweight CI install.
"""

import pytest

from llb.rag.fusion_evidence.evidence_gate import (
    READING_FLAT,
    READING_INSUFFICIENT_EVIDENCE,
    READING_SEPARATED,
    apply_evidence_gate,
    evidence_gate_note,
    evidence_gate_summary,
    minimum_discordant_pairs,
    reaches_reporting_level,
    smallest_reachable_sign_test_p,
)
from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    TIGHTER_CONFIDENCE,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    bootstrap_index_sets,
    bootstrap_interval,
    discordant_deltas,
    discordant_pairs,
    evidence_gate_clause,
    gated_readings,
    paired_comparison,
    reading_of,
    separates,
    sign_test_p,
)

RESAMPLES = 2000
SEED = 13
BOUND = minimum_discordant_pairs(DEFAULT_CONFIDENCE)


def _unanimous(wins: int, n: int = 40):
    """A paired row where `wins` items differ, ALL in the candidate's favour -- the best case.

    The most extreme arrangement there is, so if the reporting level is unreachable here it is
    unreachable for any data with that many differing items.
    """
    candidate = [1.0 if i < wins else 0.0 for i in range(n)]
    baseline = [0.0] * n
    return paired_comparison(
        candidate, baseline, bootstrap_index_sets(n, RESAMPLES, SEED), DEFAULT_CONFIDENCE
    )


# --- the bound is derived, not chosen -------------------------------------------------------


def test_the_bound_is_the_sign_tests_own_resolution_limit():
    """`2 * 0.5^d <= 1 - confidence`, evaluated -- no constant is introduced anywhere."""
    assert BOUND == 6
    assert smallest_reachable_sign_test_p(BOUND) <= 1.0 - DEFAULT_CONFIDENCE
    assert smallest_reachable_sign_test_p(BOUND - 1) > 1.0 - DEFAULT_CONFIDENCE
    # The same arithmetic the block already publishes beside its interval.
    assert sign_test_p(BOUND, 0) == pytest.approx(smallest_reachable_sign_test_p(BOUND))
    assert sign_test_p(BOUND - 1, 0) == pytest.approx(smallest_reachable_sign_test_p(BOUND - 1))


def test_a_tighter_convention_needs_more_items_and_a_looser_one_fewer():
    """The bound moves WITH the level, which is why a reading can be gated one convention up."""
    assert minimum_discordant_pairs(LOOSER_CONFIDENCE) == 5
    assert minimum_discordant_pairs(TIGHTER_CONFIDENCE) == 7
    with pytest.raises(ValueError, match="no discordant count"):
        minimum_discordant_pairs(1.0)


# --- the boundary: 5, 6, 7 differing items --------------------------------------------------


@pytest.mark.parametrize(
    ("discordant", "expected"),
    [
        (BOUND - 1, READING_INSUFFICIENT_EVIDENCE),
        (BOUND, READING_SEPARATED),
        (BOUND + 1, READING_SEPARATED),
    ],
)
def test_the_reading_flips_at_the_reachable_minimum(discordant: int, expected: str):
    row = _unanimous(discordant)
    assert row["delta"]["lo"] > 0.0  # every one of them clears zero
    assert discordant_pairs(row) == discordant
    assert reading_of(row) == expected
    assert row["stability"]["reading"] == expected
    assert separates(row) is (expected == READING_SEPARATED)


def test_the_gate_changes_the_reading_and_nothing_else():
    """The interval, the ledger, the point estimate, and the sign-test p are all untouched."""
    n = 40
    candidate = [1.0 if i < BOUND - 1 else 0.0 for i in range(n)]
    baseline = [0.0] * n
    index_sets = bootstrap_index_sets(n, RESAMPLES, SEED)
    row = paired_comparison(candidate, baseline, index_sets, DEFAULT_CONFIDENCE)
    deltas = [c - b for c, b in zip(candidate, baseline)]
    assert row["delta"] == bootstrap_interval(deltas, index_sets, DEFAULT_CONFIDENCE)
    assert (row["wins"], row["losses"], row["ties"]) == (BOUND - 1, 0, n - BOUND + 1)
    assert row["sign_test_p"] == sign_test_p(BOUND - 1, 0)
    assert row["stability"]["reading"] == READING_INSUFFICIENT_EVIDENCE
    assert row["stability"]["discordant"] == BOUND - 1


def test_a_row_at_the_bound_is_borderline_because_a_tighter_level_cannot_read_it():
    """The gate and the borderline flag are one scale: 6 items reach 95% and not 97.5%."""
    stability = _unanimous(BOUND)["stability"]
    assert stability["reading"] == READING_SEPARATED
    assert stability["tighter_reading"] == READING_INSUFFICIENT_EVIDENCE
    assert stability["borderline"] is True and stability["side"] == "above"


def test_only_a_claim_is_gated_never_the_null_reading():
    """A thin item set may say it found nothing; what it may not do is say it found something."""
    assert apply_evidence_gate(READING_FLAT, discordant=0) == READING_FLAT
    assert apply_evidence_gate(READING_SEPARATED, discordant=0) == READING_INSUFFICIENT_EVIDENCE
    assert apply_evidence_gate("neither", discordant=1, null_reading="neither") == "neither"
    assert apply_evidence_gate("answer", discordant=1, null_reading="neither") == (
        READING_INSUFFICIENT_EVIDENCE
    )
    # A row that does not clear zero reads `flat` however few items differ.
    assert reading_of(_unanimous(0)) == READING_FLAT


def test_the_two_discordant_counts_agree_on_the_same_data():
    """A lane holding delta vectors and one holding a ledger must gate identically."""
    for wins in (0, 1, BOUND - 1, BOUND, 12):
        row = _unanimous(wins)
        deltas = [1.0] * wins + [0.0] * (40 - wins)
        assert discordant_deltas(deltas) == discordant_pairs(row) == wins
        assert reaches_reporting_level(wins) is separates(row)


# --- what a report and a verdict say about it -----------------------------------------------


def test_a_report_states_how_many_readings_the_gate_relabeled():
    rows = [_unanimous(BOUND - 1), _unanimous(BOUND), _unanimous(0)]
    assert gated_readings(rows) == (1, 3)
    text = "\n".join(evidence_gate_summary(*gated_readings(rows)))
    assert "1 of 3" in text and "insufficient evidence" in text and str(BOUND) in text
    assert text.isascii()
    # Always stated, so a clean table is a statement rather than an absence.
    assert evidence_gate_summary(0, 3)
    assert evidence_gate_summary(0, 0) == []


def test_the_shared_clause_names_the_thin_rows_and_stays_quiet_otherwise():
    assert evidence_gate_note([]) == ""
    assert evidence_gate_note([("recall", BOUND)]) == ""
    clause = evidence_gate_note([("recall", BOUND - 1), ("mrr", 1)])
    assert "INSUFFICIENT EVIDENCE" in clause and "`recall` differs on 5" in clause
    assert "2 of the rows" in clause
    # A row that never cleared zero is not "insufficient evidence", it is a measured non-result.
    assert evidence_gate_clause([("recall", _unanimous(0))]) == ""
    assert "INSUFFICIENT EVIDENCE" in evidence_gate_clause([("recall", _unanimous(BOUND - 1))])


# --- the verdict guard, per lane ------------------------------------------------------------


def _lane_rows(wins: int, n: int = 40):
    """Per-item candidate/baseline vectors with `wins` unanimous differences."""
    return [1.0 if i < wins else 0.0 for i in range(n)], [0.0] * n


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_embedder_bake_off_adopts_only_on_a_reachable_reading(wins: int):
    from llb.rag.embedding_bakeoff_uncertainty import METRIC_MRR, METRIC_RECALL, paired_rows
    from llb.rag.embedding_bakeoff_verdict import DECISION_ADOPT, DECISION_RETAIN, decide_verdict

    candidate, baseline = _lane_rows(wins)
    vectors = {
        "e5": {METRIC_RECALL: baseline, METRIC_MRR: baseline},
        "bge": {METRIC_RECALL: candidate, METRIC_MRR: candidate},
    }
    verdict = decide_verdict(paired_rows(vectors, "e5", resamples=RESAMPLES, seed=SEED), "e5")
    if wins >= BOUND:
        assert verdict["decision"] == DECISION_ADOPT and verdict["model"] == "bge"
        assert "INSUFFICIENT EVIDENCE" not in verdict["reason"]
    else:
        assert verdict["decision"] == DECISION_RETAIN and verdict["model"] == "e5"
        assert "INSUFFICIENT EVIDENCE" in verdict["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_context_ablation_reads_a_thin_uplift_as_inconclusive(wins: int):
    from llb.eval.context_ablation.compare import compare_context_strategies
    from llb.eval.context_ablation.models import (
        LANE_CLOSED_BOOK,
        LANE_RAG,
        VERDICT_RAG_PAYS_OFF,
        VERDICT_RETRIEVAL_INCONCLUSIVE,
    )
    from llb.eval.context_ablation.report import format_report

    rag, closed = _lane_rows(wins)
    lanes = {
        LANE_CLOSED_BOOK: [
            {"item_id": f"q{i:02d}", "status": "ok", "objective_score": v, "retrieval_hit": 0.0}
            for i, v in enumerate(closed)
        ],
        LANE_RAG: [
            {"item_id": f"q{i:02d}", "status": "ok", "objective_score": v, "retrieval_hit": 1.0}
            for i, v in enumerate(rag)
        ],
    }
    report = compare_context_strategies(lanes, {}, resamples=RESAMPLES, seed=SEED)
    if wins >= BOUND:
        assert report["verdict"]["decision"] == VERDICT_RAG_PAYS_OFF
    else:
        assert report["verdict"]["decision"] == VERDICT_RETRIEVAL_INCONCLUSIVE
        assert "INSUFFICIENT EVIDENCE" in report["verdict"]["reason"]
        text = format_report(report)
        assert "insufficient evidence" in text and text.isascii()


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_answer_quality_lane_needs_a_reachable_objective_to_call_a_gain(wins: int):
    from llb.eval.answer_quality.compare import compare_answer_quality
    from llb.eval.answer_quality.models import VERDICT_ANSWER_GAIN, VERDICT_INCONCLUSIVE

    fused, vector = _lane_rows(wins)
    ids = [f"q{i:02d}" for i in range(len(fused))]

    def rows(values):
        return [
            {
                "item_id": item_id,
                "status": "ok",
                "objective_score": value,
                "token_f1": value,
                "exact": 0.0,
                "contains": 0.0,
                "retrieval_hit": 1.0,
            }
            for item_id, value in zip(ids, values)
        ]

    report = compare_answer_quality(
        {"vector": rows(vector), "fused/x@0.10/d10": rows(fused)},
        dict.fromkeys(ids, "multi-hop"),
        baseline="vector",
        focus_slice="multi-hop",
        resamples=RESAMPLES,
        seed=SEED,
    )
    if wins >= BOUND:
        assert report["verdict"]["decision"] == VERDICT_ANSWER_GAIN
    else:
        # The objective still LEADS, so the honest fall-through is `inconclusive`, not `no_gain`.
        assert report["verdict"]["decision"] == VERDICT_INCONCLUSIVE
        assert "INSUFFICIENT EVIDENCE" in report["verdict"]["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_adoption_bar_extends_only_on_a_reachable_cell(wins: int):
    from llb.eval.embedder_adoption.cross_model import (
        READING_ANSWER,
        cell_reading,
    )
    from llb.eval.embedder_adoption.models import (
        DECISION_EXTEND_BAR,
        DECISION_NO_EVIDENCE,
        METRIC_OBJECTIVE,
        METRIC_RECIPROCAL_RANK,
    )
    from llb.eval.embedder_adoption.verdict import decide_bar

    candidate, baseline = _lane_rows(wins)
    index_sets = bootstrap_index_sets(len(candidate), RESAMPLES, SEED)
    cell = {
        "label": "k10+rerank",
        "top_k": 10,
        "reranker": "x",
        "n": len(candidate),
        "lanes": {},
        "paired": {
            metric: paired_comparison(candidate, baseline, index_sets, DEFAULT_CONFIDENCE)
            for metric in (METRIC_OBJECTIVE, METRIC_RECIPROCAL_RANK)
        },
    }
    verdict = decide_bar([cell], baseline="e5", candidate="bge")
    if wins >= BOUND:
        assert verdict["decision"] == DECISION_EXTEND_BAR
        assert cell_reading(cell) == READING_ANSWER
    else:
        # Not `keep_bar`: the rank half is just as unreadable, so the sweep decided nothing.
        assert verdict["decision"] == DECISION_NO_EVIDENCE
        assert cell_reading(cell) == READING_INSUFFICIENT_EVIDENCE
        assert "INSUFFICIENT EVIDENCE" in verdict["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_fusion_sweep_adopts_only_on_a_reachable_focus_gain(wins: int):
    from llb.rag.fusion_evidence.models import METRICS, VERDICT_ADOPT, VERDICT_INCONCLUSIVE
    from llb.rag.fusion_evidence.slices import slice_report
    from llb.rag.fusion_evidence.verdict import decide

    fused, vector = _lane_rows(wins)
    index_sets = bootstrap_index_sets(len(fused), RESAMPLES, SEED)
    positions = list(range(len(fused)))

    def row(values):
        report = slice_report(
            dict.fromkeys(METRICS, values),
            dict.fromkeys(METRICS, vector),
            positions,
            index_sets,
            DEFAULT_CONFIDENCE,
            METRICS,
        )
        return {"overall": report, "slices": {"multi-hop": report}}

    verdict = decide(
        {"vector": row(vector), "fused/x@0.10/d10": row(fused)},
        baseline="vector",
        focus_slice="multi-hop",
    )
    if wins >= BOUND:
        assert verdict["decision"] == VERDICT_ADOPT
    else:
        assert verdict["decision"] == VERDICT_INCONCLUSIVE
        assert "INSUFFICIENT EVIDENCE" in verdict["reason"]
        assert "rests on enough differing items" in verdict["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_long_context_power_resolution_refuses_a_thin_direction(wins: int):
    from llb.eval.context_ablation.models import (
        DERIVED_LONG_CONTEXT_DELTA,
        LANE_LONG_CONTEXT,
        POWER_RESOLUTION_SEPARATED,
        POWER_RESOLUTION_UNDECIDABLE,
    )
    from llb.eval.context_ablation.power import resolve_power_analysis

    candidate, baseline = _lane_rows(wins)
    entry = {
        "label": DERIVED_LONG_CONTEXT_DELTA,
        "candidate": LANE_LONG_CONTEXT,
        "reference": "rag",
        "n": len(candidate),
        "population": "all",
        "paired": paired_comparison(
            candidate,
            baseline,
            bootstrap_index_sets(len(candidate), RESAMPLES, SEED),
            DEFAULT_CONFIDENCE,
        ),
    }
    plan = {
        "alpha": round(1.0 - DEFAULT_CONFIDENCE, 12),
        "minimum_detectable_delta": 0.01,
        "target_reached": True,
    }
    resolved = resolve_power_analysis({"derived": [entry]}, plan)  # type: ignore[arg-type]
    if wins >= BOUND:
        assert resolved["resolution"] == POWER_RESOLUTION_SEPARATED
        assert resolved["direction"] == LANE_LONG_CONTEXT
    else:
        assert resolved["resolution"] == POWER_RESOLUTION_UNDECIDABLE
        assert "differ on only" in resolved["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_routing_calibration_gate_reads_the_same_rule(wins: int):
    """The coverage half of the gate is a separation; the single-span half is not, and stays."""
    from llb.rag.fusion_evidence.stats import separates as lane_separates

    candidate, baseline = _lane_rows(wins)
    multi = paired_comparison(
        candidate,
        baseline,
        bootstrap_index_sets(len(candidate), RESAMPLES, SEED),
        DEFAULT_CONFIDENCE,
    )
    single = paired_comparison(
        baseline, baseline, bootstrap_index_sets(len(baseline), RESAMPLES, SEED), DEFAULT_CONFIDENCE
    )
    gate = lane_separates(multi, DEFAULT_CONFIDENCE) and single["delta"]["lo"] >= 0.0
    assert gate is (wins >= BOUND)
