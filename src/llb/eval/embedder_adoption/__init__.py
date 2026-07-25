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
from llb.eval.embedder_adoption.cross_model import (
    READING_ANSWER,
    READING_NEITHER,
    READING_RANK_ONLY,
    cell_reading,
    compare_models,
    format_cross_model,
    format_cross_summary,
)
from llb.eval.embedder_adoption.models import (
    DECISION_EXTEND_BAR,
    DECISION_KEEP_BAR,
    DECISION_NO_EVIDENCE,
    CellSpec,
    EmbedderLane,
)
from llb.eval.embedder_adoption.report import format_report, format_summary
from llb.eval.embedder_adoption.roster_report import format_roster, format_roster_summary
from llb.eval.embedder_adoption.roster import (
    DECISION_INSUFFICIENT_VARIATION,
    DECISION_NO_PROPERTY_PREDICTS,
    DECISION_PROPERTY_PREDICTS,
    DEFAULT_FOCUS_CELL,
    ModelProfile,
    compare_roster,
    decide_roster,
)
from llb.eval.embedder_adoption.screen import (
    DECISION_FULL_SET_REQUIRED,
    DECISION_SCREEN_SUPPORTED,
    DEFAULT_SCREEN_SIZES,
    cell_item_deltas,
    decide_screen,
    format_screen,
    format_screen_summary,
    reading_from_deltas,
    run_screen_study,
    screen_model,
)
from llb.eval.embedder_adoption.run import (
    load_profiles,
    load_report,
    run_adoption_bar_sweep,
    run_cross_model_comparison,
    run_roster_comparison,
    run_screen_study_over_paths,
)

__all__ = [
    "DECISION_EXTEND_BAR",
    "DECISION_FULL_SET_REQUIRED",
    "DECISION_INSUFFICIENT_VARIATION",
    "DECISION_KEEP_BAR",
    "DECISION_NO_EVIDENCE",
    "DECISION_NO_PROPERTY_PREDICTS",
    "DECISION_PROPERTY_PREDICTS",
    "DECISION_SCREEN_SUPPORTED",
    "DEFAULT_FOCUS_CELL",
    "DEFAULT_SCREEN_SIZES",
    "READING_ANSWER",
    "READING_NEITHER",
    "READING_RANK_ONLY",
    "CellSpec",
    "EmbedderLane",
    "ModelProfile",
    "build_cells",
    "cell_config",
    "cell_item_deltas",
    "cell_reading",
    "compare_cells",
    "compare_models",
    "compare_roster",
    "decide_bar",
    "decide_roster",
    "decide_screen",
    "format_cross_model",
    "format_cross_summary",
    "format_report",
    "format_roster",
    "format_roster_summary",
    "format_screen",
    "format_screen_summary",
    "format_summary",
    "load_profiles",
    "load_report",
    "parse_rerankers",
    "parse_top_ks",
    "reading_from_deltas",
    "run_adoption_bar_sweep",
    "run_cross_model_comparison",
    "run_roster_comparison",
    "run_screen_study",
    "run_screen_study_over_paths",
    "screen_model",
    "with_reciprocal_rank",
]
