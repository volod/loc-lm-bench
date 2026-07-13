"""Operator recommendation summary from final-split run bundles.

Turns the ranked leaderboard into a few plain-language picks an operator actually needs after a
sweep: the best RAG accuracy, the most efficient model for THIS host (quality per watt), the fastest,
and the model we recommend running here -- the highest-accuracy candidate that is feasible,
Pareto-optimal, and fits the GPU tier's VRAM budget with headroom. Selection is pure and testable;
host detection and chart rendering live behind injectable seams / a guarded matplotlib import.

The implementation is split into `model` (constants + dataclasses + formatting primitives), `build`
(bundle loading + cohort selection + `build_recommendation`), `render` (the summary markdown +
payload + config-detail table), and `sections` (the miss / self-improvement / fine-tune / context
sections). The public API is re-exported here so callers keep importing `llb.board.recommend`.
"""

from llb.board.recommend.build import (
    build_recommendation,
    load_config_cells,
    load_run_summaries,
    select_cohort,
)
from llb.board.recommend.model import (
    MISS_SECTION_MAX_RECOMMENDATIONS,
    RAG_CONFIG_KEYS,
    SAFE_VRAM_FRACTION,
    HostInfo,
    Recommendation,
    RunSummary,
    _short,
)
from llb.board.recommend.render import (
    format_config_detail_md,
    format_summary_md,
    recommendation_payload,
)
from llb.board.recommend.sections import (
    format_chain_context_section_md,
    format_finetune_campaign_section_md,
    format_miss_section_md,
    format_self_improvement_section_md,
    latest_chain_context,
    latest_finetune_campaign,
    latest_self_improvement,
)

__all__ = [
    "MISS_SECTION_MAX_RECOMMENDATIONS",
    "RAG_CONFIG_KEYS",
    "SAFE_VRAM_FRACTION",
    "HostInfo",
    "Recommendation",
    "RunSummary",
    "_short",
    "build_recommendation",
    "format_chain_context_section_md",
    "format_config_detail_md",
    "format_finetune_campaign_section_md",
    "format_miss_section_md",
    "format_self_improvement_section_md",
    "format_summary_md",
    "latest_chain_context",
    "latest_finetune_campaign",
    "latest_self_improvement",
    "load_config_cells",
    "load_run_summaries",
    "recommendation_payload",
    "select_cohort",
]
