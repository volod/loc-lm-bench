"""Reason clauses and final verdict projection for embedding bake-offs."""

from collections.abc import Sequence

from llb.rag.embedding_bakeoff_uncertainty import (
    BARS,
    METRIC_MRR,
    METRIC_RECALL,
    BakeoffVerdict,
    PairedRow,
    bar_stability,
)
from llb.rag.fusion_evidence.paired import evidence_gate_clause
from llb.rag.fusion_evidence.selection import SelectionAdjustment
from llb.rag.fusion_evidence.stability import borderline_note
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, format_interval


def gate_note(
    paired: dict[str, PairedRow],
    baseline: str,
    bars: Sequence[str],
    confidence: float = DEFAULT_CONFIDENCE,
) -> str:
    return evidence_gate_clause(
        [
            (f"{model} {bar}", row["metrics"][bar])
            for model, row in sorted(paired.items())
            if model != baseline
            for bar in BARS
            if bar in bars
        ],
        confidence,
    )


def near_miss_note(paired: dict[str, PairedRow], borderline: dict[str, list[str]]) -> str:
    return borderline_note(
        [
            (f"{model} {bar}", bar_stability(paired[model], bar))
            for model, marked in borderline.items()
            for bar in marked
        ]
    )


def rank_key(paired: PairedRow, cleared: Sequence[str]) -> tuple[int, float, float]:
    return (
        -len(cleared),
        -paired["metrics"][METRIC_RECALL]["delta"]["mean"],
        -paired["metrics"][METRIC_MRR]["delta"]["mean"],
    )


def adopt_reason(
    winner: str,
    baseline: str,
    paired: PairedRow,
    cleared: Sequence[str],
) -> str:
    detail = "; ".join(
        f"{bar} delta {format_interval(paired['metrics'][bar]['delta'])}, "
        f"{paired['metrics'][bar]['wins']}/{paired['metrics'][bar]['losses']}/"
        f"{paired['metrics'][bar]['ties']} win/loss/tie, "
        f"sign-test p={paired['metrics'][bar]['sign_test_p']:.3f}"
        for bar in cleared
    )
    return f"`{winner}` clears {', '.join(cleared)} against `{baseline}`: {detail}"


def build_verdict(
    decision: str,
    model: str | None,
    baseline: str | None,
    bars: Sequence[str],
    *,
    separated: list[str] | None = None,
    cleared: dict[str, list[str]] | None = None,
    borderline: dict[str, list[str]] | None = None,
    per_row_cleared: dict[str, list[str]] | None = None,
    adjustment: SelectionAdjustment | None = None,
    reason: str,
) -> BakeoffVerdict:
    verdict: BakeoffVerdict = {
        "decision": decision,
        "model": model,
        "baseline": baseline,
        "separated": list(separated or []),
        "bars": list(bars),
        "cleared": dict(cleared or {}),
        "borderline": dict(borderline or {}),
        "reason": reason,
    }
    if per_row_cleared is not None:
        verdict["per_row_cleared"] = per_row_cleared
    if adjustment is not None:
        verdict["selection_adjustment"] = adjustment
    return verdict
