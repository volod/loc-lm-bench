"""The adopt-or-negative rule and the report it renders, over synthetic comparison reports."""

from llb.graph.resolution.overlay import NodeCluster, NodeOverlay
from llb.graph.resolution.report import format_console_summary, format_resolution_report
from llb.graph.resolution.verdict import (
    DECISION_NEGATIVE,
    DECISION_RECOMMEND,
    baseline_rows,
    decide,
    threshold_rows,
)

STRATEGY = "local_khop"
BASE = f"graph/{STRATEGY}"
OVERLAY = f"graph/{STRATEGY}+overlay@0.9"


def _report(overlay_recall: float, decision: str, lane: str) -> dict:
    return {
        "k": 10,
        "n": 5,
        "backends": {
            BASE: {"n": 5, "k": 10, "recall_at_k": 0.5, "mrr": 0.25},
            OVERLAY: {"n": 5, "k": 10, "recall_at_k": overlay_recall, "mrr": 0.3},
            "faiss": {"n": 5, "k": 10, "recall_at_k": 0.9, "mrr": 0.8},
        },
        "best_recall": lane,
        "paired_items": [],
        "uncertainty": {
            "baseline": BASE,
            "eligible_lanes": [BASE, OVERLAY],
            "resamples": 0,
            "confidence": 0.95,
            "seed": 7,
        },
        "verdict": {
            "decision": decision,
            "lane": lane,
            "baseline": BASE,
            "reason": f"`{lane}` reason",
        },
    }


def _overlay() -> NodeOverlay:
    return NodeOverlay(threshold=0.9, clusters=(NodeCluster(canonical_id=1, member_ids=(0, 1)),))


def test_a_separated_overlay_lane_is_recommended_and_named():
    verdict = decide({STRATEGY: _report(0.7, "adopt", OVERLAY)})
    assert verdict["decision"] == DECISION_RECOMMEND
    assert verdict["lane"] == OVERLAY
    assert "no shipped store is rewritten" in verdict["note"]


def test_an_adopted_baseline_is_not_an_adopted_overlay():
    verdict = decide({STRATEGY: _report(0.5, "adopt", BASE)})
    assert verdict["decision"] == DECISION_NEGATIVE
    assert verdict["lane"] is None


def test_the_negative_reason_names_every_strategy_that_was_read():
    verdict = decide(
        {
            STRATEGY: _report(0.4, "retain", BASE),
            "global_community": _report(0.4, "retain", BASE),
        }
    )
    assert verdict["decision"] == DECISION_NEGATIVE
    assert "local_khop:" in verdict["reason"] and "global_community:" in verdict["reason"]


def test_one_strategy_separating_is_enough_to_recommend():
    verdict = decide(
        {
            STRATEGY: _report(0.4, "retain", BASE),
            "global_community": _report(0.8, "adopt", "graph/global_community+overlay@0.9"),
        }
    )
    assert verdict["decision"] == DECISION_RECOMMEND
    assert verdict["strategy"] == "global_community"


def test_threshold_rows_carry_what_was_merged_and_the_delta_against_the_baseline():
    rows = threshold_rows({STRATEGY: _report(0.7, "adopt", OVERLAY)}, [_overlay()])
    assert rows[0]["n_nodes_merged"] == 1
    scored = rows[0]["strategies"][STRATEGY]
    assert scored["lane"] == OVERLAY
    assert scored["recall_at_k"] == 0.7
    assert scored["delta_recall_at_k"] == 0.7 - 0.5


def test_baseline_rows_read_the_pre_overlay_lane():
    assert baseline_rows({STRATEGY: _report(0.7, "adopt", OVERLAY)}) == {
        STRATEGY: {"lane": BASE, "recall_at_k": 0.5, "mrr": 0.25}
    }


def _summary(decision: str = "adopt") -> dict:
    reports = {STRATEGY: _report(0.7, decision, OVERLAY if decision == "adopt" else BASE)}
    overlays = [_overlay()]
    return {
        "mode": "graph-node-overlay",
        "n_nodes": 30,
        "n_edges": 10,
        "n_items": 5,
        "k": 10,
        "linkage": {
            "n_scored_pairs": 94,
            "n_matched_pairs": 8,
            "match_threshold": 0.6,
            "seed": 7,
            "untrained_levels": ["name/Exact match on name"],
        },
        "baseline": baseline_rows(reports),
        "thresholds": threshold_rows(reports, overlays),
        "verdict": decide(reports),
    }


def test_the_markdown_report_states_the_merge_counts_and_the_verdict():
    rendered = format_resolution_report(_summary())
    assert "| 0.9 | 1 | 1 | 2 | local_khop | 0.700 | +0.200 |" in rendered
    assert "Recommended overlay lane" in rendered
    assert "unreachable by construction" in rendered


def test_the_markdown_report_names_a_negative_result_as_the_result():
    rendered = format_resolution_report(_summary("retain"))
    assert "Negative result: no overlay is adopted" in rendered


def test_a_declined_run_renders_its_reason_in_both_forms():
    summary = {"declined": True, "reason": "too few nodes", "n_nodes": 3}
    assert "too few nodes" in format_resolution_report(summary)
    assert format_console_summary(summary) == "[graph-resolution] not run: too few nodes"


def test_the_console_summary_prints_a_line_per_cut_and_strategy():
    printed = format_console_summary(_summary())
    assert "cut 0.9: 1 cluster(s), 1 node(s) merged, largest 2" in printed
    assert "local_khop: recall_at_k 0.700 (+0.200), mrr 0.300 (+0.050)" in printed
