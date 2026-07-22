"""RAG-versus-long-context ablation: does retrieval pay for itself on this corpus?

A leaderboard row says how well a model answers WITH retrieval; it never says how much of that
score retrieval bought. Three lanes over the identical item set answer that -- `closed_book` (no
context at all), `rag` (the run configuration as-is), and `long_context` (the item's whole source
document laid into the prompt, skipped rather than truncated when it does not fit) -- and the
report states retrieval uplift, the long-context delta, and the per-item contamination flag.

The lanes are DIAGNOSTIC: `rag` stays the leaderboard row, and nothing here changes a ranking.
"""

from llb.eval.context_ablation.compare import compare_context_strategies
from llb.eval.context_ablation.lanes import default_lanes, lane_config, parse_lanes
from llb.eval.context_ablation.models import LANES
from llb.eval.context_ablation.report import format_report
from llb.eval.context_ablation.run import run_context_ablation

__all__ = [
    "LANES",
    "compare_context_strategies",
    "default_lanes",
    "format_report",
    "lane_config",
    "parse_lanes",
    "run_context_ablation",
]
