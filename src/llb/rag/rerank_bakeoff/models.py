"""Contracts of the reranker bake-off: the roster, the loaded-scorer seam, and the report rows.

Kept apart from the scoring lane (`llb.rag.rerank_bakeoff.lane`) for the same reason
`embedding_bakeoff_models.py` is: the row shape is read by the scorer, the renderer, and the CLI,
while the load/score orchestration is read by none of them.

A reranker row carries THREE kinds of fact, and a reranker is chosen on all three:

  - **rank quality** -- recall@k, MRR, and the mean first-hit rank the reranker delivers over a
    FIXED candidate pool (same encoder, same chunking, same pool for every candidate);
  - **cost** -- mean rerank wall-clock per query and the VRAM the model holds, at rest and at its
    scoring peak, because the second model has to fit beside the generator;
  - **paired uncertainty** -- the delta against the incumbent reranker over the shared item set,
    which is what says whether a rank lead is a ranking or an item set.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from typing_extensions import NotRequired, TypedDict

from llb.rag.candidate_screen import SkippedCandidate
from llb.rag.card_parity import CardParityResult
from llb.rag.embedding_bakeoff_uncertainty import (
    BakeoffVerdict,
    PairedRow,
    UncertaintySettings,
)
from llb.rag.rerank import RerankScorer

if TYPE_CHECKING:  # imported lazily: the floor is opt-in and costs one extra pass
    from llb.rag.noise_floor_models import NoiseFloorReport

# The incumbent every candidate is PAIRED against: the pinned `DEFAULT_RERANKER`. A swap
# recommendation is a statement about replacing THIS model, so it is the natural baseline.
DEFAULT_BASELINE_RERANKER = "BAAI/bge-reranker-v2-m3"

# The reranker-OFF row. Not a candidate that can be downloaded: it is the retrieval order the pool
# arrived in, scored by the identical metric, so "is the second model worth it at all?" is a row in
# the table rather than a separate run.
ROW_NO_RERANK = "none"

KIND_RERANK = "rerank"
KIND_RETRIEVAL_ORDER = "retrieval_order"

# Default candidates: the incumbent plus the multilingual cross-encoders an operator would
# shortlist today. Two need `trust_remote_code` and are skipped unless opted into; the last two are
# the current decoder-based generation, which is exactly where the VRAM column earns its place.
DEFAULT_RERANK_CANDIDATES_ROSTER = [
    DEFAULT_BASELINE_RERANKER,
    "jinaai/jina-reranker-v2-base-multilingual",  # trust_remote_code
    "Alibaba-NLP/gte-multilingual-reranker-base",  # trust_remote_code
    "mixedbread-ai/mxbai-rerank-base-v2",
    "Qwen/Qwen3-Reranker-0.6B",
]

# Why a roster entry produced no row. `SKIP_REMOTE_CODE` (imported from the embedder lane) is a
# policy decline; these two are host facts, and both carry the measurement that produced them.
SKIP_NO_HEADROOM = "vram_headroom_exceeded"
SKIP_LOAD_FAILED = "load_failed"

MB = 1024 * 1024


@dataclass
class LoadedScorer:
    """One loaded reranker plus what holding it costs, behind the injectable loader seam.

    `scorer` is the `RerankScorer` the lane calls; `vram_mb` is the resident footprint measured at
    load (None when no GPU reader is available); `release` frees the weights so the next candidate's
    footprint is its own rather than the roster's running total.
    """

    scorer: RerankScorer
    device: str | None = None
    load_seconds: float = 0.0
    vram_mb: float | None = None
    # Current footprint on demand, so the lane can read the PEAK after a scoring pass rather than
    # trusting the at-rest number a batch of long passages can exceed.
    read_vram: Callable[[], float | None] = lambda: None
    release: Callable[[], None] = lambda: None


# model id -> LoadedScorer. The CLI binds the real cross-encoder loader; tests inject fakes.
ScorerLoader = Callable[[str], LoadedScorer]


class ScorerLoadError(RuntimeError):
    """A candidate that could not be loaded on this host (download, OOM, missing extra).

    Raised by the loader seam and turned into a recorded `skipped` row: a candidate the host cannot
    run is a fact about the host, and the rest of the roster still ranks.
    """


class VramHeadroom(TypedDict):
    """The VRAM budget a candidate has to fit into beside the resident generator.

    `headroom_mb` is what the reranker may hold: total device memory minus the generator's declared
    residency minus the reserve. It is None when the operator declared no generator, in which case
    the footprint columns are reported and the fit gate does not run -- a measurement without a
    budget is not a verdict.
    """

    total_mb: float | None
    generator_mb: float | None
    reserve_mb: float
    headroom_mb: float | None


class RerankCandidateResult(TypedDict):
    """One reranker's row: rank quality over the shared pool, plus what it costs to keep it."""

    model: str
    kind: str
    family: str
    recall_at_k: float
    mrr: float
    # Mean rank of the FIRST hit among items that have one inside k (None when nothing hits), plus
    # the count behind that mean -- a mean rank over a shrinking denominator is not comparable
    # without it.
    first_hit_rank_mean: float | None
    hit_items: int
    n: int
    k: int
    pool_depth: int
    rerank_ms_per_query: float
    pairs_per_second: float
    load_seconds: NotRequired[float]
    device: NotRequired[str]
    vram_mb: NotRequired[float | None]  # resident footprint at load
    vram_peak_mb: NotRequired[float | None]  # peak observed across the scoring pass
    fits_headroom: NotRequired[bool]  # peak footprint inside the declared budget
    trust_remote_code: NotRequired[bool]
    default_prompt: NotRequired[str]
    # Did this candidate reproduce its own model card before it was ranked? A row scored without
    # the check says so (`no_reference_declared`) rather than reading as verified.
    card_parity: NotRequired[CardParityResult]
    # Paired percentile-bootstrap delta against the incumbent reranker over shared resample index
    # sets -- the reading that says whether a point-estimate lead is a ranking or an item set.
    paired_vs_baseline: NotRequired[PairedRow]


class RerankBakeoffReport(TypedDict):
    """The bake-off artifact: the ranked rows, how they were produced, and the keep-or-swap call."""

    k: int
    n: int
    corpus_root: str
    # The retrieval configuration every candidate reranked, pinned so two runs are comparable.
    embedding_model: str
    chunking: str
    pool_depth: int
    batch_size: int
    # The load precision every candidate was held at (`auto` keeps each checkpoint's own), so two
    # runs' latency and VRAM columns are comparable only when this line matches.
    dtype: NotRequired[str]
    candidates: list[RerankCandidateResult]
    best_recall: str | None
    best_first_hit: str | None
    # Roster entries that produced no row (declined, unloadable, or over the VRAM budget).
    skipped: NotRequired[list[SkippedCandidate]]
    headroom: NotRequired[VramHeadroom]
    uncertainty: NotRequired[UncertaintySettings]
    # Keep-or-swap. Point-estimate rank order is NOT the recommendation; this is.
    verdict: NotRequired[BakeoffVerdict]
    paired_items: NotRequired[list[dict[str, Any]]]
    # Measurement floor over the RERANK scores (opt-in). See `llb.rag.rerank_bakeoff.lane`.
    noise_floor: NotRequired["NoiseFloorReport"]
