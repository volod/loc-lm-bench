"""RAG-versus-long-context ablation: does retrieval pay for itself, and how far can it go?

A leaderboard row says how well a model answers WITH retrieval; it never says how much of that
score retrieval bought. Four lanes over the identical item set answer that -- `closed_book` (no
context at all), `rag` (the run configuration as-is), `retrieved_document` (retrieve as
configured, then send the whole document the top-ranked chunk came from), and `long_context` (the
item's whole GOLD document). Both document lanes skip rather than truncate an item that does not
fit. The report states retrieval uplift, the oracle long-context delta, the SPLIT of that delta
into a capturable part and a pure-oracle part, and the per-item contamination flag.

`closed_book` and `long_context` are DIAGNOSTIC -- `rag` stays the leaderboard row and nothing
here changes a ranking. `retrieved_document` is the one shippable configuration among them, so it
carries its own adopt-or-reject verdict.
"""

from llb.eval.context_ablation.compare import compare_context_strategies
from llb.eval.context_ablation.lanes import default_lanes, lane_config, parse_lanes
from llb.eval.context_ablation.models import LANES
from llb.eval.context_ablation.report import format_report
from llb.eval.context_ablation.run import run_context_ablation
from llb.eval.context_ablation.verdict_adoption import decide_retrieved_document

__all__ = [
    "LANES",
    "compare_context_strategies",
    "decide_retrieved_document",
    "default_lanes",
    "format_report",
    "lane_config",
    "parse_lanes",
    "run_context_ablation",
]
