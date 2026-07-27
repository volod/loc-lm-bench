"""Types and vocabulary of the embedder adoption-bar sweep.

The embedder bake-off adopts on recall@k alone, and a candidate can be separated on the OTHER
metric: an encoder that ranks the same evidence earlier without finding more of it gains MRR while
its recall delta spans zero. Whether that is worth adopting is a fact the retrieval table cannot
see, because it depends on the SCORED configuration: at a small `top_k`, or under a cross-encoder
reranker that can only re-sort what it is handed, first-hit rank is the binding constraint; at a
generous `top_k` with room in the context budget, ranking earlier changes nothing the answer reads.

This lane measures it END TO END. Each CELL is one retrieval configuration (`top_k` x reranker);
inside a cell the two encoders are scored on the IDENTICAL items and paired against each other, so
the answer-side delta per cell is what decides whether the bake-off gets a second, scoped adoption
bar (`llb.rag.embedding_bakeoff_uncertainty.BAR_FIRST_HIT`) or keeps recall@k as its sole bar.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from typing_extensions import NotRequired, TypedDict

from llb.eval.answer_quality.models import (
    METRIC_OBJECTIVE,
    METRIC_RETRIEVAL_HIT,
    METRIC_TOKEN_F1,
)
from llb.rag.fusion_evidence.stats import Interval
from llb.rag.fusion_evidence.paired import PairedComparison

if TYPE_CHECKING:
    from llb.rag.fusion_evidence.stability import (
        ReadingStability as RowStability,
    )

# Per-case columns compared inside every cell. `objective_score` is the decision metric.
# `contains` is the verbosity-robust found-rate companion: token F1 mixes finding the needle with
# being terse, so a verbose model reads as weak on the objective alone.
# `retrieval_hit` is the any-span coverage signal at the cell's own `top_k` -- the half of the
# effect that is retrieval, not generation.
# `reciprocal_rank` is derived from the bundle's `first_hit_rank` column: it is FIRST-HIT RANK
# itself, measured inside the scored configuration, so the cell can show whether the encoder's
# ranking advantage even survives the reranker before asking whether it reached the answer.
METRIC_CONTAINS = "contains"
METRIC_RECIPROCAL_RANK = "reciprocal_rank"
CELL_METRICS = (
    METRIC_OBJECTIVE,
    METRIC_CONTAINS,
    METRIC_TOKEN_F1,
    METRIC_RETRIEVAL_HIT,
    METRIC_RECIPROCAL_RANK,
)

# The per-case column `reciprocal_rank` is derived from (`llb.executor.cases`).
COLUMN_FIRST_HIT_RANK = "first_hit_rank"

# Cell label markers: `k10`, `k3+rerank`.
CELL_RERANK_MARKER = "+rerank"

# The cell the model-dependence question is about: a generous `top_k` WITH the cross-encoder,
# where first-hit rank is binding but recall is already at ceiling, so a gain there is the purest
# "the reranker made the encoder's ranking pay" signal.
DEFAULT_FOCUS_CELL = "k10+rerank"

# Sweep decisions -- what the measurement says about the bake-off's adoption bar.
# The rank gain reached the answers in at least one cell: the scoped first-hit bar is justified.
DECISION_EXTEND_BAR = "extend_bar"
# The rank gain reproduces in the scored configurations but never reaches the answers: recall@k
# stays the sole bar.
DECISION_KEEP_BAR = "keep_bar"
# The candidate's first-hit-rank advantage does not reproduce in ANY scored cell, so this sweep
# cannot decide the bar either way -- the premise it rests on was not met.
DECISION_NO_EVIDENCE = "no_evidence"


class EmbedderLane(NamedTuple):
    """One encoder and the data root whose `llb/rag` store was built with it.

    A store is embedded and queried by ONE encoder (`run-eval` refuses a mismatch), so an encoder
    lane is inseparable from its store location; carrying them together is what lets one sweep
    score two encoders without rebuilding an index per cell.
    """

    model: str
    data_dir: Path


class CellSpec(NamedTuple):
    """One retrieval configuration the two encoders are compared inside."""

    top_k: int
    reranker: str | None

    @property
    def label(self) -> str:
        """`k10` / `k3+rerank` -- stable, sortable, and the key of every artifact."""
        return f"k{self.top_k}{CELL_RERANK_MARKER if self.reranker else ''}"


@dataclass(frozen=True)
class ItemDeltas:
    """Per-item candidate-minus-baseline deltas of one cell, aligned across the two encoders.

    A dataclass rather than a mapping: the two vectors are always read together and always by name,
    and it keeps a subsample from being built with mismatched lengths. The sweep builds it from the
    vectors it already holds in memory; the screen and roster re-derive it from the run bundles a
    finished sweep names, and both orders are the sorted shared item ids, so the two agree exactly.
    """

    item_ids: list[str]
    objective: list[float]
    reciprocal_rank: list[float]

    def __post_init__(self) -> None:
        if not len(self.item_ids) == len(self.objective) == len(self.reciprocal_rank):
            raise ValueError("item ids, objective, and reciprocal-rank deltas must align")

    def __len__(self) -> int:
        return len(self.item_ids)

    def take(self, indexes: Sequence[int]) -> "ItemDeltas":
        """The same deltas restricted to `indexes` -- one screen-sized subsample."""
        return ItemDeltas(
            [self.item_ids[i] for i in indexes],
            [self.objective[i] for i in indexes],
            [self.reciprocal_rank[i] for i in indexes],
        )


class LaneMetrics(TypedDict):
    """One encoder's per-metric means inside one cell, with bootstrap bounds."""

    run_dirs: list[str]
    metrics: dict[str, Interval]


class CellReport(TypedDict):
    """One cell: both encoders' means plus the candidate-minus-baseline paired deltas."""

    label: str
    top_k: int
    reranker: str | None
    n: int
    lanes: dict[str, LaneMetrics]
    paired: dict[str, PairedComparison]
    # How settled this cell's binary reading is (`llb.eval.embedder_adoption.stability`), measured
    # from the SAME per-item deltas and the SAME resample draw the intervals above came from.
    # Absent only when the sweep drew no resamples, where an exceedance probability has no meaning.
    stability: NotRequired["RowStability"]


class BarVerdict(TypedDict):
    """Keep-or-extend: the one sentence the adoption bar is changed (or not changed) on."""

    decision: str  # DECISION_EXTEND_BAR | DECISION_KEEP_BAR | DECISION_NO_EVIDENCE
    baseline: str
    candidate: str
    # Cells whose paired OBJECTIVE delta clears zero -- the scope a second bar would carry.
    answer_cells: list[str]
    # Cells whose paired RECIPROCAL-RANK delta clears zero -- where the premise holds at all.
    rank_cells: list[str]
    # Cells a neighbouring conventional confidence level would read differently. A qualifier on the
    # cells above, never a fourth outcome: a borderline cell still counts where it counted.
    borderline_cells: list[str]
    reason: str


class AdoptionBarReport(TypedDict):
    """The full sweep artifact: every cell, the ranking premise, and the bar verdict."""

    baseline: str
    candidate: str
    item_ids: list[str]
    metrics: list[str]
    resamples: int
    confidence: float
    seed: int
    cells: list[CellReport]
    verdict: BarVerdict
    metadata: NotRequired[dict[str, object]]
