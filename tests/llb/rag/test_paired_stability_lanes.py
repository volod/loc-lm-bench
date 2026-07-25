"""adoption-borderline-annotation-on-the-other-paired-lanes -- every lane qualifies its verdict.

The annotation itself rides on `paired_comparison` (see `test_paired_stability.py`), so what needs
covering per lane is the SENTENCE: each adopt-or-retain reason must say when the row it just named
sits on the cut, and each report must render the boundary table. One test per lane, each driving
the lane's own public comparison over a delta tuned to land just below the 95% cut.

Pure: value vectors and dict rows, so the whole vertical runs in the lightweight CI install.
"""

from llb.rag.fusion_evidence.stats import bootstrap_index_sets

RESAMPLES = 2000
SEED = 13


def _near_miss(n: int = 30, wins: int = 6, losses: int = 1) -> tuple[list[float], list[float]]:
    """Candidate/baseline vectors whose paired delta lands just BELOW the 95% cut."""
    return (
        [1.0 if i < wins else 0.0 for i in range(n)],
        [1.0 if wins <= i < wins + losses else 0.0 for i in range(n)],
    )


def test_the_embedder_bake_off_retain_says_when_a_candidate_missed_by_nothing():
    """A `retain` reached by a knife-edge miss must not print like one reached by a mile."""
    from llb.rag.embedding_bakeoff_uncertainty import METRIC_MRR, METRIC_RECALL, paired_rows
    from llb.rag.embedding_bakeoff_verdict import DECISION_RETAIN, decide_verdict

    candidate, baseline = _near_miss()
    vectors = {
        "e5": {METRIC_RECALL: baseline, METRIC_MRR: baseline},
        "bge": {METRIC_RECALL: candidate, METRIC_MRR: candidate},
    }
    rows = paired_rows(vectors, "e5", resamples=RESAMPLES, seed=SEED)
    verdict = decide_verdict(rows, "e5")
    assert verdict["decision"] == DECISION_RETAIN  # the rule itself never moves
    assert verdict["borderline"] == {"bge": [METRIC_RECALL]}
    assert "BORDERLINE" in verdict["reason"] and "`bge recall_at_k`" in verdict["reason"]

    settled = {
        "e5": {METRIC_RECALL: [0.0] * 30, METRIC_MRR: [0.0] * 30},
        "bge": {METRIC_RECALL: [0.0] * 30, METRIC_MRR: [0.0] * 30},
    }
    flat = decide_verdict(paired_rows(settled, "e5", resamples=RESAMPLES, seed=SEED), "e5")
    assert flat["decision"] == DECISION_RETAIN
    assert flat["borderline"] == {} and "BORDERLINE" not in flat["reason"]


def test_the_context_ablation_verdict_qualifies_a_knife_edge_uplift():
    from llb.eval.context_ablation.compare import compare_context_strategies
    from llb.eval.context_ablation.models import LANE_CLOSED_BOOK, LANE_RAG

    rag, closed = _near_miss()
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
    reason = report["verdict"]["reason"]
    assert "BORDERLINE" in reason and "retrieval_uplift" in reason
    assert "### How close each derived delta sits to the cut" in _ablation_text(report)


def _ablation_text(report) -> str:
    from llb.eval.context_ablation.report import format_report

    text = format_report(report)
    assert text.isascii()
    return text


def test_the_answer_quality_verdict_qualifies_a_knife_edge_focus_slice():
    from llb.eval.answer_quality.compare import compare_answer_quality

    fused, vector = _near_miss()
    ids = [f"q{i:02d}" for i in range(len(fused))]

    def rows(values, hit):
        return [
            {
                "item_id": item_id,
                "status": "ok",
                "objective_score": value,
                "token_f1": value,
                "exact": 0.0,
                "contains": 0.0,
                "retrieval_hit": hit,
            }
            for item_id, value in zip(ids, values)
        ]

    report = compare_answer_quality(
        {"vector": rows(vector, 1.0), "fused/x@0.10/d10": rows(fused, 1.0)},
        dict.fromkeys(ids, "multi-hop"),
        baseline="vector",
        focus_slice="multi-hop",
        resamples=RESAMPLES,
        seed=SEED,
    )
    assert "BORDERLINE" in report["verdict"]["reason"]
    assert "objective" in report["verdict"]["reason"]

    from llb.eval.answer_quality.report import format_report

    text = format_report(report)
    assert "### How close each lane sits to the cut (multi-hop)" in text
    assert text.isascii()


def test_the_fusion_sweep_verdict_qualifies_a_knife_edge_focus_gain():
    """A fused row that misses the cut by nothing must not read like one that does not help."""
    from llb.rag.fusion_evidence.models import METRICS
    from llb.rag.fusion_evidence.verdict import decide
    from llb.rag.fusion_evidence.slices import slice_report

    fused, vector = _near_miss()
    n = len(fused)
    index_sets = bootstrap_index_sets(n, RESAMPLES, SEED)
    positions = list(range(n))

    def row(values):
        vectors = dict.fromkeys(METRICS, values)
        base = dict.fromkeys(METRICS, vector)
        report = slice_report(vectors, base, positions, index_sets, 0.95, METRICS)
        return {"overall": report, "slices": {"multi-hop": report}}

    verdict = decide(
        {"vector": row(vector), "fused/x@0.10/d10": row(fused)},
        baseline="vector",
        focus_slice="multi-hop",
    )
    assert verdict["decision"] == "inconclusive"  # the rule itself never moves
    assert "BORDERLINE" in verdict["reason"]
    assert "`recall_at_k`" in verdict["reason"] or "`all_spans_at_k`" in verdict["reason"]
