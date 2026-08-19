"""Reranker bake-off: rank cross-encoders on ONE accepted ledger at a FIXED encoder and chunking.

The shipped cross-encoder has been pinned to one model since the seam was built and has never been
compared with anything, while the adoption evidence shows the reranked cell is where a retrieval
change actually reaches the answer. A reranker is also the cheapest place to buy first-hit rank on a
16 GiB host -- and the cheapest place to lose the VRAM the generator needs. So the choice is made
here on evidence: rank quality AND cost, on the same item set, against the incumbent.

The design that makes the reading a statement about RERANKERS: the candidate pool is retrieved ONCE
(one encoder, one chunking, one depth) and every candidate re-sorts that identical pool. The
reranker-off row rides along as the pool's own retrieval order, so "is the second model worth it at
all?" is a row in the same table rather than a separate run.

Every candidate is also PAIRED against the incumbent over shared resample index sets, and the run
ends in a keep-or-swap verdict a lead inside its own sampling interval cannot win -- the same
machinery the embedder bake-off reads (`llb.rag.embedding_bakeoff_uncertainty`), because the
statistics take metric vectors, not encoders.

Pure and injectable: the model loader is a seam, so scoring, ranking, the fit gate, the paired
verdict, and the report all run over a fake cross-encoder -- no download, no GPU.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from llb.rag.candidate_screen import SkippedCandidate
from llb.rag.card_parity import blocks_scoring, parity_skip_row
from llb.rag.embedding_bakeoff import paired_item_ledger
from llb.rag.embedding_bakeoff_models import BakeoffItem
from llb.rag.embedding_bakeoff_uncertainty import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    METRIC_MRR,
    METRIC_RECALL,
    MetricVectors,
    item_vectors,
)
from llb.rag.rerank_bakeoff.cards import check_rerank_card
from llb.rag.rerank_bakeoff.families import resolve_convention
from llb.rag.rerank_bakeoff.readings import attach_uncertainty, measure_rerank_floor
from llb.rag.rerank_bakeoff.fit import (
    fit_verdict,
    load_candidate,
    peak_footprint,
    skip_row,
)
from llb.rag.rerank_bakeoff.models import (
    DEFAULT_BASELINE_RERANKER,
    KIND_RETRIEVAL_ORDER,
    ROW_NO_RERANK,
    SKIP_LOAD_FAILED,
    RerankBakeoffReport,
    RerankCandidateResult,
    ScorerLoadError,
    ScorerLoader,
    VramHeadroom,
)
from llb.rag.rerank_bakeoff.scoring import (
    CandidatePass,
    Pools,
    candidate_row,
    rerank_pass,
    retrieval_order_pass,
    scored_pairs,
)

_LOG = logging.getLogger(__name__)

# The reranker lane is BY CONSTRUCTION the configuration where first-hit rank binds: a cross-encoder
# can only re-sort the pool it is handed, so rank is the quantity it moves. That is exactly the
# scope the embedder lane declares for its second adoption bar
# (`docs/impl/current/rag-core/first-hit-rank-adoption.md`), so here BOTH bars are on by default.
DEFAULT_RERANK_BARS = (METRIC_RECALL, METRIC_MRR)


@dataclass(slots=True)
class _ScoredRerankers:
    """What one bake-off accumulates per candidate: its row, its per-item vectors, its floor pools.

    A record rather than three parallel dicts: the floor is measured over the SAME passes after
    every candidate is scored, so the three have to stay aligned.
    """

    k: int
    pool_depth: int
    items: list[BakeoffItem]
    rows: list[RerankCandidateResult] = field(default_factory=list)
    vectors: dict[str, MetricVectors] = field(default_factory=dict)
    passes: dict[str, CandidatePass] = field(default_factory=dict)

    def record(self, model: str, finished: CandidatePass, **row_kwargs: Any) -> None:
        """Shape the row and per-item vectors of one finished pass and keep the pass for the floor."""
        pairs = scored_pairs(finished.ranked, self.items, self.k)
        self.rows.append(
            candidate_row(
                model,
                pairs,
                k=self.k,
                pool_depth=self.pool_depth,
                seconds=finished.seconds,
                **row_kwargs,
            )
        )
        self.vectors[model] = item_vectors(pairs, self.k)
        self.passes[model] = finished


def best_by(rows: list[RerankCandidateResult], metric: str) -> str | None:
    """Point-estimate leader on `metric`; ties break by the other metric, then by rerank latency."""
    if not rows:
        return None
    other = METRIC_MRR if metric == METRIC_RECALL else METRIC_RECALL
    best = min(
        rows,
        key=lambda row: (-row[metric], -row[other], row["rerank_ms_per_query"], row["model"]),  # type: ignore[literal-required]
    )
    return best["model"]


def run_rerank_bakeoff(
    items: list[BakeoffItem],
    pools: Pools,
    k: int,
    *,
    corpus_root: str,
    embedding_model: str,
    chunking: str,
    pool_depth: int,
    batch_size: int,
    candidates: Sequence[str],
    load_scorer: ScorerLoader,
    dtype: str | None = None,
    item_ids: Sequence[str] | None = None,
    skipped: Sequence[SkippedCandidate] = (),
    headroom: VramHeadroom | None = None,
    baseline: str | None = DEFAULT_BASELINE_RERANKER,
    bars: Sequence[str] = DEFAULT_RERANK_BARS,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    noise_floor: bool = False,
    noise_floor_replicates: int | None = None,
) -> RerankBakeoffReport:
    """Score the reranker-off row plus every loadable candidate over the SHARED pools, then rank.

    `pools` is one candidate pool per item in retrieval order, retrieved once at `pool_depth`; each
    candidate re-sorts it, so the rows differ only by the cross-encoder. A candidate that cannot be
    loaded, whose resident footprint exceeds the declared headroom, or that does not reproduce its
    own model card lands in `skipped` WITH its measurement rather than vanishing from the table.

    The card check runs on the LOADED scorer, before the pass: loading is not evidence a candidate
    can be ranked, reproducing its card is (`llb.rag.rerank_bakeoff.cards`).
    """
    if len(pools) != len(items):
        raise ValueError("the reranker bake-off needs one candidate pool per scored item")
    if item_ids is not None and len(item_ids) != len(items):
        raise ValueError("the reranker paired ledger needs one item id per scored item")
    scored = _ScoredRerankers(k=k, pool_depth=pool_depth, items=items)
    declined = list(skipped)

    scored.record(ROW_NO_RERANK, retrieval_order_pass(pools), kind=KIND_RETRIEVAL_ORDER)
    for model in candidates:
        _LOG.info("[compare-rerankers] loading candidate: %s", model)
        loaded, refusal = load_candidate(model, load_scorer, headroom)
        if loaded is None:
            declined.append(refusal)  # type: ignore[arg-type]
            continue
        parity = check_rerank_card(model, loaded.scorer)
        if blocks_scoring(parity):
            _LOG.warning("[compare-rerankers] %s failed card parity: %s", model, parity["detail"])
            declined.append(parity_skip_row(parity, resolve_convention(model).family))
            loaded.release()
            continue
        try:
            finished = rerank_pass(pools, items, loaded.scorer, label=model)
        except ScorerLoadError as exc:
            # A candidate that loads and then dies mid-pass (OOM on a long passage, a device-side
            # assert inside its own kernels) is the same kind of fact as one that never loaded:
            # recorded with what the host said, not a hole in the table.
            _LOG.warning("[compare-rerankers] %s failed while scoring: %s", model, exc)
            declined.append(skip_row(model, SKIP_LOAD_FAILED, f"failed while scoring: {exc}"))
            loaded.release()
            continue
        # NVML after the pass, not during it: the caching allocator keeps what the pass reserved,
        # so the reading after the last batch IS the peak the generator has to live beside.
        peak = peak_footprint(loaded)
        scored.record(
            model,
            finished,
            loaded=loaded,
            peak_vram_mb=peak,
            fits_headroom=fit_verdict(peak, headroom),
            card_parity=parity,
        )
        loaded.release()

    report: RerankBakeoffReport = {
        "k": k,
        "n": len(items),
        "corpus_root": corpus_root,
        "embedding_model": embedding_model,
        "chunking": chunking,
        "pool_depth": pool_depth,
        "batch_size": batch_size,
        "candidates": scored.rows,
        "best_recall": best_by(scored.rows, METRIC_RECALL),
        "best_first_hit": best_by(scored.rows, METRIC_MRR),
        "paired_items": paired_item_ledger(scored.vectors, len(items), item_ids),
    }
    if dtype is not None:
        report["dtype"] = dtype
    if declined:
        report["skipped"] = declined
    if headroom is not None:
        report["headroom"] = headroom
    attach_uncertainty(
        report,
        scored.vectors,
        baseline,
        bars=bars,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    if noise_floor:
        report["noise_floor"] = measure_rerank_floor(
            scored.passes,
            items,
            k,
            pool_depth,
            replicates=noise_floor_replicates,
            seed=seed,
        )
    return report
