"""Vocabulary of the operator recommendation: the VRAM/RAG constants, the report-template helper
`_t`, the three record dataclasses (`RunSummary`, `HostInfo`, `Recommendation`), and the small pure
formatting primitives (`_short`, `_vram`, `_qpw`, `_md_table`, `_fmt_float`, `_float_for_sort`).

A leaf module -- `build` (selection), `render` (summary markdown), and `sections` (extra sections)
build on it.
"""

import logging
from dataclasses import dataclass

from llb.board.runs import RunRecord
from llb.core.contracts.results import BoardRow
from llb.core.contracts.common import JsonObject
from llb.prompts.registry import render_text
from llb.scoring.leaderboard import ModelResult

_LOG = logging.getLogger(__name__)

# Keep some VRAM headroom so the "recommended for this host" pick is not a card pinned at 100%.
SAFE_VRAM_FRACTION = 0.92
RAG_CONFIG_KEYS = ("strategy", "chunk_size", "chunk_overlap", "top_k", "retrieval_mode")
# The recommend summary quotes at most this many ranked miss-analysis recommendation lines;
# the full ranked list stays in the analysis report it links.
MISS_SECTION_MAX_RECOMMENDATIONS = 5


def _t(name: str, **values: object) -> str:
    """Render a `board.recommend.<name>` text template. The report prose lives in prompt templates
    (`prompts/templates/board/recommend/`) so the wording is reviewable in files, not inline here."""
    return render_text(f"board.recommend.{name}", values)


@dataclass
class RunSummary:
    """One model's best final-split run plus the host-efficiency + retrieval fields the board omits."""

    record: RunRecord
    quality_per_watt: float | None
    mean_power_w: float | None
    recall_at_k: float | None
    mrr: float | None

    @property
    def result(self) -> ModelResult:
        return self.record.result

    @property
    def model(self) -> str:
        return self.record.result.model


@dataclass
class HostInfo:
    tier_gb: int
    total_mb: int
    gpu_name: str
    detected: bool


@dataclass
class Recommendation:
    host: HostInfo
    summaries: list[RunSummary]  # the ranked cohort (shared split + n_cases)
    excluded: list[RunSummary]  # off-cohort runs named but not ranked (different split/n_cases)
    ranked: list[BoardRow]
    policy: str
    best_quality: RunSummary
    best_efficiency: RunSummary | None
    fastest: RunSummary
    recommended_for_host: RunSummary
    recall_at_k: float | None
    mrr: float | None
    top_k: int | None
    rag_config: JsonObject
    min_tokens_per_s: float = (
        0.0  # good-enough-performance floor applied to the host pick (0 = off)
    )


def _short(model: str) -> str:
    """A compact label for a model: the last path/tag segment, trimmed of a GGUF quant suffix."""
    tail = model.rstrip("/").split("/")[-1]
    return tail


def _vram(summary: RunSummary) -> str:
    vram = summary.result.peak_vram_mb
    return "n/a" if vram is None else f"{vram:.0f} MiB"


def _qpw(summary: RunSummary | None) -> str:
    if summary is None or summary.quality_per_watt is None:
        return "n/a"
    return f"{summary.quality_per_watt:.3f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _float_for_sort(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("-inf")


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.4f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
