"""Logging-based presentation for completed evaluation runs."""

import logging
from pathlib import Path

from llb.config import RunConfig
from llb.contracts import RetrievalMetrics, RunPaths, TelemetryReport

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
) -> None:
    """Log the concise user-facing summary for one completed run."""
    _LOG.info("[run-eval] model=%s backend=%s cases=%d", config.model, config.backend, n_cases)
    _LOG.info(
        "[run-eval] retrieval: recall@%d=%.3f mrr=%.3f",
        config.top_k,
        retrieval_metrics["recall_at_k"],
        retrieval_metrics["mrr"],
    )
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
