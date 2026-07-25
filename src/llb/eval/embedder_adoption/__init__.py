"""Does an encoder that only RANKS better deserve a second adoption bar? Measured end to end.

The embedder bake-off adopts on recall@k alone. A candidate separated on MRR but not on recall
ranks the same evidence earlier without finding more of it, and whether that is worth its cost
depends on a downstream fact the retrieval table cannot see: at a small `top_k`, or under a
cross-encoder reranker that only re-sorts what it is given, first-hit rank is the binding
constraint; at a generous `top_k` it is nearly free. This lane scores the identical item set end to
end in every (`top_k` x reranker) cell under both encoders and reports the answer-side delta per
cell, so the bake-off's adoption bar is kept or extended on measurement rather than on argument.
"""

from llb.eval.embedder_adoption.cells import build_cells, cell_config, parse_rerankers, parse_top_ks
from llb.eval.embedder_adoption.compare import compare_cells, decide_bar, with_reciprocal_rank
from llb.eval.embedder_adoption.models import (
    DECISION_EXTEND_BAR,
    DECISION_KEEP_BAR,
    DECISION_NO_EVIDENCE,
    CellSpec,
    EmbedderLane,
)
from llb.eval.embedder_adoption.report import format_report, format_summary
from llb.eval.embedder_adoption.run import run_adoption_bar_sweep

__all__ = [
    "DECISION_EXTEND_BAR",
    "DECISION_KEEP_BAR",
    "DECISION_NO_EVIDENCE",
    "CellSpec",
    "EmbedderLane",
    "build_cells",
    "cell_config",
    "compare_cells",
    "decide_bar",
    "format_report",
    "format_summary",
    "parse_rerankers",
    "parse_top_ks",
    "run_adoption_bar_sweep",
    "with_reciprocal_rank",
]
