"""The two READINGS a reranker bake-off ends in: the paired keep-or-swap, and the noise floor.

Split from the lane (`llb.rag.rerank_bakeoff.lane`) along the same seam the embedder bake-off uses:
building and scoring the candidates is one job, deciding what the resulting numbers SUPPORT is
another, and only the second one is statistics.

Both readings are shared machinery pointed at reranker rows: the paired intervals and the verdict
come from `llb.rag.embedding_bakeoff_uncertainty` (they take metric vectors, not encoders), and the
floor comes from `llb.rag.noise_floor` -- read here on each lane's OWN rerank scores at a
scale-matched jitter, because two cross-encoder heads do not share a score scale.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from llb.rag.embedding_bakeoff_selection import adjust_bakeoff_selection
from llb.rag.embedding_bakeoff_uncertainty import MetricVectors, paired_rows
from llb.rag.embedding_bakeoff_verdict import decide_verdict
from llb.rag.rerank_bakeoff.models import RerankBakeoffReport
from llb.rag.rerank_bakeoff.scoring import (
    CandidatePass,
    floor_pools,
    score_scale,
    scored_pairs,
)

if TYPE_CHECKING:
    from llb.rag.embedding_bakeoff_models import BakeoffItem
    from llb.rag.noise_floor_models import NoiseFloorReport

# The score key the reranker writes on every kept chunk (`llb.rag.rerank.rerank_chunks`).
RERANK_SCORE_KEY = "rerank_score"


def attach_uncertainty(
    report: RerankBakeoffReport,
    vectors: dict[str, MetricVectors],
    baseline: str | None,
    *,
    bars: Sequence[str],
    resamples: int,
    confidence: float,
    seed: int,
) -> None:
    """Hang the paired interval on each row and the keep-or-swap verdict on the report.

    A baseline the run did not score (the incumbent declined, or unloadable on this host) leaves the
    rows bare and the verdict `undecided` rather than silently re-pointing the comparison at
    whichever candidate happened to rank first.
    """
    report["uncertainty"] = {
        "baseline": baseline,
        "bars": list(bars),
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
    }
    paired = (
        paired_rows(vectors, baseline, resamples=resamples, confidence=confidence, seed=seed)
        if baseline is not None
        else {}
    )
    for row in report["candidates"]:
        if row["model"] in paired:
            row["paired_vs_baseline"] = paired[row["model"]]
    report["verdict"] = decide_verdict(
        paired,
        baseline,
        bars,
        confidence,
        adjustment=adjust_bakeoff_selection(
            vectors, baseline, bars, resamples=resamples, seed=seed
        ),
    )


def measure_rerank_floor(
    passes: dict[str, CandidatePass],
    items: list["BakeoffItem"],
    k: int,
    pool_depth: int,
    *,
    replicates: int | None = None,
    seed: int,
) -> "NoiseFloorReport":
    """The measurement floor over each lane's OWN ranking scores, at scale-matched amplitudes.

    Every lane is perturbed at `DEFAULT_SCORE_JITTER` scaled by its median within-pool score range,
    so a model whose head simply emits bigger numbers does not get a proportionally tighter floor.
    """
    from llb.rag.noise_floor import DEFAULT_REPLICATES, DEFAULT_SCORE_JITTER, measure_pool_floor
    from llb.rag.retrieval import evaluate_retrieval

    lane_pools = {
        model: floor_pools(finished.ranked, items, RERANK_SCORE_KEY)
        for model, finished in passes.items()
    }
    bases = {
        model: evaluate_retrieval(scored_pairs(finished.ranked, items, k), k)
        for model, finished in passes.items()
    }
    return measure_pool_floor(
        lane_pools,
        bases,
        k,
        candidates=pool_depth,
        replicates=replicates or DEFAULT_REPLICATES,
        seed=seed,
        jitter_by_lane={
            model: DEFAULT_SCORE_JITTER * score_scale(pools) for model, pools in lane_pools.items()
        },
    )
