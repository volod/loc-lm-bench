"""Content recommendations from a `MissAnalysis` and the composed, ranked recommendation list:
add prompt-system dictionary terms, try the named alternative model, and review the
refusal / artifact / judge-disagreement piles. Every line names its numeric evidence.

`build_recommendations` composes these with the retrieval-depth advice in `rec_retrieval.py`,
heaviest-miss-pile first; `analyze_run` calls it, and `refresh_recommendations` rebuilds after
probe outcomes attach.
"""

import json
from collections import Counter
from pathlib import Path

from llb.board.miss_analysis.model import (
    DICTIONARY_CLUSTER_MIN,
    DICTIONARY_CLUSTER_SHARE,
    JUDGE_AGREEMENT_MIN,
    MISS_ARTIFACT,
    MISS_GENERATION,
    MISS_JUDGE,
    MISS_REFUSAL,
    MissAnalysis,
    MissRecord,
    _t,
)
from llb.board.miss_analysis.rec_retrieval import (
    _lower_top_k_recommendation,
    _retrieval_recommendations,
)
from llb.core.contracts.common import JsonObject


def _generation_recommendations(
    analysis: MissAnalysis, alternatives: list[tuple[str, float]]
) -> list[JsonObject]:
    n_generation = analysis.class_counts[MISS_GENERATION]
    if not n_generation:
        return []
    recs: list[JsonObject] = []
    generation_misses = [m for m in analysis.misses if m.miss_class == MISS_GENERATION]
    cluster = _dominant_generation_cluster(analysis, generation_misses)
    if cluster is not None:
        dimension, key, count = cluster
        recs.append(
            {
                "action": "dictionary_terms",
                "weight": count,
                "line": _t(
                    "rec_dictionary",
                    cluster=key,
                    dimension=dimension,
                    n_cluster=count,
                    n_generation=n_generation,
                ),
            }
        )
    better = [(m, obj) for m, obj in alternatives if m != analysis.model]
    if better:
        alt_model, alt_objective = max(better, key=lambda pair: pair[1])
        run_objective = _run_mean_objective(analysis)
        if alt_objective > run_objective:
            recs.append(
                {
                    "action": "alternative_model",
                    "weight": n_generation,
                    "line": _t(
                        "rec_alternative_model",
                        alt_model=alt_model,
                        alt_objective=f"{alt_objective:.3f}",
                        objective=f"{run_objective:.3f}",
                        model=analysis.model,
                        split=analysis.split,
                        n_generation=n_generation,
                    ),
                }
            )
    return recs


def _dominant_generation_cluster(
    analysis: MissAnalysis, generation_misses: list[MissRecord]
) -> tuple[str, str, int] | None:
    """The (dimension, key, count) of the densest document/topic cluster of generation misses,
    when it is big enough to suggest missing domain vocabulary."""
    best: tuple[str, str, int] | None = None
    for dimension, attribute in (("document", "source_doc_id"), ("topic", "topic")):
        counts = Counter(getattr(m, attribute) for m in generation_misses)
        if not counts:
            continue
        key, count = counts.most_common(1)[0]
        if count >= DICTIONARY_CLUSTER_MIN and count / len(generation_misses) >= (
            DICTIONARY_CLUSTER_SHARE
        ):
            if best is None or count > best[2]:
                best = (dimension, key, count)
    return best


def _run_mean_objective(analysis: MissAnalysis) -> float:
    """Mean objective of the analyzed run, recovered from its manifest via the bundle path."""
    try:
        manifest = json.loads(
            (Path(analysis.run_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        return float((manifest.get("metrics") or {}).get("objective_score", 0.0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def _status_recommendations(analysis: MissAnalysis) -> list[JsonObject]:
    recs: list[JsonObject] = []
    n_refusal = analysis.class_counts[MISS_REFUSAL]
    if n_refusal:
        recs.append(
            {
                "action": "refusal_review",
                "weight": n_refusal,
                "line": _t("rec_refusals", n_refusal=n_refusal, n_misses=len(analysis.misses)),
            }
        )
    n_artifact = analysis.class_counts[MISS_ARTIFACT]
    if n_artifact:
        breakdown = Counter(m.status for m in analysis.misses if m.miss_class == MISS_ARTIFACT)
        recs.append(
            {
                "action": "artifact_review",
                "weight": n_artifact,
                "line": _t(
                    "rec_artifacts",
                    n_artifact=n_artifact,
                    breakdown=", ".join(f"{status}={n}" for status, n in sorted(breakdown.items())),
                ),
            }
        )
    n_judge = analysis.class_counts[MISS_JUDGE]
    if n_judge:
        recs.append(
            {
                "action": "judge_review",
                "weight": n_judge,
                "line": _t(
                    "rec_judge",
                    n_judge=n_judge,
                    judge_min=JUDGE_AGREEMENT_MIN,
                    threshold=analysis.threshold,
                ),
            }
        )
    return recs


def build_recommendations(
    analysis: MissAnalysis, *, alternatives: list[tuple[str, float]] | None = None
) -> list[JsonObject]:
    """Ranked, evidence-backed recommendation lines; heaviest miss pile first."""
    recs = (
        _retrieval_recommendations(analysis)
        + _lower_top_k_recommendation(analysis)
        + _generation_recommendations(analysis, alternatives or [])
        + _status_recommendations(analysis)
    )
    return sorted(recs, key=lambda rec: int(rec["weight"]), reverse=True)


def refresh_recommendations(
    analysis: MissAnalysis, *, alternatives: list[tuple[str, float]] | None = None
) -> None:
    """Rebuild the ranked recommendations (call after attaching probe outcomes)."""
    analysis.recommendations = build_recommendations(analysis, alternatives=alternatives)
