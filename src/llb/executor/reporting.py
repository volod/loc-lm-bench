"""Logging-based presentation for completed evaluation runs."""

import logging
from pathlib import Path

from llb.core.config import RunConfig
from llb.core.contracts.rag import RetrievalMetrics
from llb.core.contracts.runs import RunMetrics, RunPaths
from llb.core.contracts.hardware import TelemetryReport

_LOG = logging.getLogger(__name__)


def _measurement(value: object, unit: str = "") -> str:
    return "n/a" if value is None else f"{value}{unit}"


def emit_summary(
    config: RunConfig,
    n_cases: int,
    retrieval_metrics: RetrievalMetrics,
    table: str,
    telemetry: TelemetryReport | None,
    paths: RunPaths,
    worksheet: Path | str | None,
    worksheet_rows: int,
    metrics: RunMetrics | None = None,
) -> None:
    """Log the concise user-facing summary for one completed run."""
    _LOG.info("[run-eval] model=%s backend=%s cases=%d", config.model, config.backend, n_cases)
    _LOG.info(
        "[run-eval] retrieval: recall@%d=%.3f mrr=%.3f",
        config.top_k,
        retrieval_metrics["recall_at_k"],
        retrieval_metrics["mrr"],
    )
    _emit_answer_side(metrics)
    _LOG.info("\n%s", table)
    if telemetry:
        _LOG.info(
            "[run-eval] telemetry: %s tok/s (load %s, peak VRAM %s, served ctx %s)",
            telemetry["steady_tokens_per_s"],
            _measurement(telemetry["load_time_s"], "s"),
            _measurement(telemetry["peak_vram_mb"], " MB"),
            _measurement(telemetry["served_context"]),
        )
    _LOG.info("[run-eval] manifest -> %s", paths["manifest"])
    _LOG.info("[run-eval] scores   -> %s (mirror: %s)", paths["scores"], paths["mirror"])
    if worksheet is not None:
        _LOG.info(
            "[run-eval] worksheet -> %s (%d rows; add human_rating)",
            worksheet,
            worksheet_rows,
        )


def _emit_answer_side(metrics: RunMetrics | None) -> None:
    """Log the answer-side RAG-quality line when any signal is present (groundedness-citation-metrics)."""
    if metrics is None:
        return
    parts: list[str] = []
    if "groundedness" in metrics:
        parts.append(f"groundedness={metrics['groundedness']:.3f}")
    if "citation_validity" in metrics:
        parts.append(f"citation_validity={metrics['citation_validity']:.3f}")
    if "hallucinated_citation_rate" in metrics:
        parts.append(f"hallucinated_citations={metrics['hallucinated_citation_rate']:.3f}")
    if "abstention_accuracy" in metrics:
        parts.append(
            f"abstention_acc={metrics['abstention_accuracy']:.3f} (n={metrics.get('n_probes', 0)})"
        )
    if parts:
        _LOG.info("[run-eval] answer-side: %s", " ".join(parts))
