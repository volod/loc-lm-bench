"""re-decide-the-relabeled-fusion-and-bakeoff-readings -- a withdrawn reading says what it needs.

The minimum-evidence gate turns a claim that rests on too few differing items into
`insufficient_evidence`. That is a statement about the ITEM SET, not about the difference, so a
recommendation an operator may still be acting on must not be left reading like a measured tie.
Every withdrawn row is therefore priced from its own discordance rate: `d` differing items out of
`n` extrapolate to the item count at which the reporting level becomes reachable at all.

Covered: the arithmetic and its two boundaries (already reachable, nothing differing), that the
floor moves with the reporting convention exactly as the bound it inverts does, the shared clause
every lane appends, the per-row column in the boundary table, and the recorded open questions whose
prices the current docs quote -- including the encoder row, which no committed goldset can reach.

Pure: value vectors and dict rows, so the whole vertical runs in the lightweight CI install.
"""

from pathlib import Path

from llb.rag.fusion_evidence.evidence_gate import (
    resolving_item_count,
)


from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    bootstrap_index_sets,
)


from llb.rag.fusion_evidence.paired import (
    paired_comparison,
)


from tests.llb.rag._paired_open_question_helpers import (
    RESAMPLES,
    SEED,
    BOUND,
)


def test_the_bake_off_retain_names_the_item_count_its_adopt_would_need():
    from llb.rag.embedding_bakeoff.uncertainty import METRIC_MRR, METRIC_RECALL, paired_rows
    from llb.rag.embedding_bakeoff.verdict import DECISION_RETAIN, decide_verdict

    n = 250
    candidate = [1.0 if i < BOUND - 1 else 0.0 for i in range(n)]
    baseline = [0.0] * n
    vectors = {
        "e5-base": {METRIC_RECALL: baseline, METRIC_MRR: baseline},
        "e5-large": {METRIC_RECALL: candidate, METRIC_MRR: candidate},
    }
    verdict = decide_verdict(
        paired_rows(vectors, "e5-base", resamples=RESAMPLES, seed=SEED), "e5-base"
    )
    assert verdict["decision"] == DECISION_RETAIN
    assert "OPEN QUESTION" in verdict["reason"]
    assert f"{BOUND - 1} of {n} -> 300 items" in verdict["reason"]


def test_the_fusion_inconclusive_names_the_focus_slice_size_it_would_need():
    from llb.rag.fusion_evidence.models import METRICS, VERDICT_INCONCLUSIVE
    from llb.rag.fusion_evidence.slices import slice_report
    from llb.rag.fusion_evidence.verdict import decide

    n = 35
    fused = [1.0 if i < 4 else 0.0 for i in range(n)]
    vector = [0.0] * n
    index_sets = bootstrap_index_sets(n, RESAMPLES, SEED)
    positions = list(range(n))

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
        {"vector": row(vector), "fused/x@0.30/d50/ioverlap": row(fused)},
        baseline="vector",
        focus_slice="multi-hop",
    )
    assert verdict["decision"] == VERDICT_INCONCLUSIVE
    assert "4 of 35 -> 53 items" in verdict["reason"]


def test_the_long_context_resolution_prices_its_undecidable_direction():
    from llb.eval.context_ablation.models import (
        DERIVED_LONG_CONTEXT_DELTA,
        LANE_LONG_CONTEXT,
        LANE_RAG,
        POWER_RESOLUTION_UNDECIDABLE,
    )
    from llb.rag.fusion_evidence.power import plan_from_deltas, resolve_power_analysis

    n = 82
    candidate = [1.0 if i < BOUND - 1 else 0.0 for i in range(n)]
    baseline = [0.0] * n
    deltas = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(candidate, baseline)
    ]
    entry = {
        "label": DERIVED_LONG_CONTEXT_DELTA,
        "candidate": LANE_LONG_CONTEXT,
        "reference": LANE_RAG,
        "n": n,
        "population": "all",
        "paired": paired_comparison(
            candidate,
            baseline,
            bootstrap_index_sets(n, RESAMPLES, SEED),
            DEFAULT_CONFIDENCE,
        ),
    }
    plan = plan_from_deltas(
        Path("reference.json"),
        deltas,
        minimum_detectable_delta=0.01,
        target_power=0.8,
        confidence=DEFAULT_CONFIDENCE,
        planned_n=n,
        selector={
            "lane": "compare-context-strategies",
            "candidate": LANE_LONG_CONTEXT,
            "baseline": LANE_RAG,
            "metric": "objective_score",
            "population": "all",
        },
    )
    resolved = resolve_power_analysis(
        plan,
        deltas,
        entry["paired"],
        candidate=LANE_LONG_CONTEXT,
        baseline=LANE_RAG,
    )
    assert resolved["resolution"] == POWER_RESOLUTION_UNDECIDABLE
    assert (
        f"realized discordance floor is {resolving_item_count(BOUND - 1, n)} items"
        in resolved["reason"]
    )
