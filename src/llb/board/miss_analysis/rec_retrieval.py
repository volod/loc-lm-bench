"""Retrieval-depth recommendations from a `MissAnalysis`: raise `top_k` / change chunking when
retrieval misses dominate, and lower `top_k` when a shallower probe measurably helped.

These read `analysis.probes` (attached by the bounded probe mode) to cite measured recovery; the
content-side advice (dictionary terms, alternative model, status review) lives in
`recommendations.py`, which composes both.
"""

from llb.board.miss_analysis.model import (
    MISS_RETRIEVAL,
    PROBE_CONFIRM_MIN,
    PROBE_MIN_OBJECTIVE_GAIN,
    MissAnalysis,
    _t,
)
from llb.core.contracts import JsonObject


def _probe_note(analysis: MissAnalysis) -> tuple[str, JsonObject | None]:
    """The raise-top_k evidence fragment from the deepest above-current probe, when one ran."""
    top_k = analysis.rag_config.get("top_k") or 0
    deeper = [p for p in analysis.probes if int(p["top_k"]) > int(top_k)]
    if not deeper:
        return "", None
    probe = max(deeper, key=lambda p: int(p["top_k"]))
    n_retrieval = int(probe.get("n_retrieval_misses", 0))
    recovered = int(probe.get("recovered_retrieval", 0))
    if n_retrieval and recovered / n_retrieval >= PROBE_CONFIRM_MIN:
        note = _t("probe_confirmed", k=probe["top_k"], recovered=recovered, n=n_retrieval)
    else:
        note = _t("probe_rejected", k=probe["top_k"], recovered=recovered, n=n_retrieval)
    return note, probe


def _retrieval_recommendations(analysis: MissAnalysis) -> list[JsonObject]:
    n_retrieval = analysis.class_counts[MISS_RETRIEVAL]
    if not n_retrieval:
        return []
    recs: list[JsonObject] = []
    top_k = analysis.rag_config.get("top_k")
    note, probe = _probe_note(analysis)
    recs.append(
        {
            "action": "raise_top_k",
            "weight": n_retrieval,
            "line": _t(
                "rec_raise_top_k",
                top_k=top_k,
                n_retrieval=n_retrieval,
                n_misses=len(analysis.misses),
                probe_note=note,
            ),
        }
    )
    if probe is not None:
        unrecovered = int(probe["n_retrieval_misses"]) - int(probe["recovered_retrieval"])
        if unrecovered > 0:
            recs.append(
                {
                    "action": "change_chunking",
                    "weight": unrecovered,
                    "line": _t(
                        "rec_chunking_probed",
                        strategy=analysis.rag_config.get("strategy"),
                        size=analysis.rag_config.get("chunk_size"),
                        overlap=analysis.rag_config.get("chunk_overlap"),
                        n_unrecovered=unrecovered,
                        n_retrieval=probe["n_retrieval_misses"],
                        max_k=probe["top_k"],
                    ),
                }
            )
    else:
        recs.append(
            {
                "action": "change_chunking",
                "weight": n_retrieval,
                "line": _t(
                    "rec_chunking_unprobed",
                    strategy=analysis.rag_config.get("strategy"),
                    size=analysis.rag_config.get("chunk_size"),
                    overlap=analysis.rag_config.get("chunk_overlap"),
                    n_retrieval=n_retrieval,
                    top_k=top_k,
                ),
            }
        )
    return recs


def _lower_top_k_recommendation(analysis: MissAnalysis) -> list[JsonObject]:
    """Recommend a SHALLOWER context only when a below-current probe measurably beat the miss
    subset's original mean objective (fewer distractors helped)."""
    top_k = analysis.rag_config.get("top_k") or 0
    shallower = [p for p in analysis.probes if int(p["top_k"]) < int(top_k)]
    if not shallower:
        return []
    best = max(shallower, key=lambda p: float(p.get("mean_objective", 0.0)))
    base = float(best.get("base_mean_objective", 0.0))
    gain = float(best.get("mean_objective", 0.0)) - base
    if gain < PROBE_MIN_OBJECTIVE_GAIN:
        return []
    return [
        {
            "action": "lower_top_k",
            "weight": len(analysis.misses),
            "line": _t(
                "rec_lower_top_k",
                k=best["top_k"],
                top_k=top_k,
                probe_obj=f"{float(best['mean_objective']):.3f}",
                base_obj=f"{base:.3f}",
            ),
        }
    ]
