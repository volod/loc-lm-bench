"""Metrics for the runner: aggregate one ranked leaderboard row from the scored cases, attach the
answer-side + stage-latency signals, and collect optional backend telemetry.

`run_eval` calls `_aggregate` (rows + metrics) and `_collect_optional_telemetry`; the NVML VRAM
reader lives with the backend readers in `runner_backend.py`.
"""

from collections.abc import Mapping

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.core.contracts import CaseScoreRow, LeaderboardRow, RunMetrics, TelemetryReport
from llb.eval import common as eval_common
from llb.executor.runner_backend import _vram_reader
from llb.scoring.aggregate import ModelResult, rank_results
from llb.scoring.judge import judge_is_trusted


def _aggregate(
    config: RunConfig,
    case_rows: list[CaseScoreRow],
    judge_rho: float | None,
    telemetry: Mapping[str, object],
    judge_score: float | None = None,
) -> tuple[list[LeaderboardRow], RunMetrics]:
    n = len(case_rows)
    objective = sum(r["objective_score"] for r in case_rows) / n if n else 0.0
    ok = [r for r in case_rows if r["status"] == eval_common.OK]
    reliability = len(ok) / n if n else 0.0
    tok_rates = [r["tokens_per_s"] for r in ok if r["tokens_per_s"] > 0]
    observed_tokens_per_s = sum(tok_rates) / len(tok_rates) if tok_rates else 0.0
    steady_rate = telemetry.get("steady_tokens_per_s")
    tokens_per_s = (
        float(steady_rate)
        if isinstance(steady_rate, int | float) and steady_rate > 0
        else observed_tokens_per_s
    )
    peak_vram = telemetry.get("peak_vram_mb")
    result = ModelResult(
        model=config.model,
        backend=config.backend,
        objective_score=objective,
        n_cases=n,
        reliability=reliability,
        tokens_per_s=tokens_per_s,
        peak_vram_mb=float(peak_vram) if isinstance(peak_vram, int | float) else None,
        judge_score=judge_score,
        feasible=True,
    )
    # The judge is trusted only when calibrated AND it actually produced a score this run.
    trusted = judge_is_trusted(judge_rho, config.judge_threshold) and judge_score is not None
    rows = rank_results([result], judge_trusted=trusted)
    metrics: RunMetrics = {
        "objective_score": objective,
        "reliability": reliability,
        "tokens_per_s": tokens_per_s,
    }
    mean_power = telemetry.get("mean_power_w")
    if isinstance(mean_power, int | float) and mean_power > 0:
        metrics["mean_power_w"] = round(float(mean_power), 2)
        metrics["tokens_per_watt"] = round(tokens_per_s / float(mean_power), 4)
        metrics["quality_per_watt"] = round(objective * tokens_per_s / float(mean_power), 4)
    stage = _stage_latency(case_rows)
    if stage:
        metrics["stage_latency"] = stage
    if judge_score is not None:
        metrics["judge_score"] = round(judge_score, 4)
    _attach_answer_side_metrics(metrics, case_rows)
    return rows, metrics


def _attach_answer_side_metrics(metrics: RunMetrics, case_rows: list[CaseScoreRow]) -> None:
    """Mean per-case groundedness / citation signals (groundedness-citation-metrics), when present."""
    for key in (
        "groundedness",
        "citation_validity",
        "citation_coverage",
        "hallucinated_citation_rate",
    ):
        values = [float(row[key]) for row in case_rows if key in row]
        if values:
            metrics[key] = round(sum(values) / len(values), 4)


def _stage_latency(case_rows: list[CaseScoreRow]) -> dict[str, float]:
    """Mean per-case stage wall-clock (rerank-context-order): retrieve / rerank / generate.

    Retrieve and rerank means cover the cases that recorded them (rerank only exists when a
    reranker is configured); generate is the mean backend latency. Empty when nothing was
    measured, so pre-existing bundles keep their shape."""

    def mean_of(key: str) -> float | None:
        values = [float(row[key]) for row in case_rows if key in row]  # type: ignore[literal-required]
        return round(sum(values) / len(values), 4) if values else None

    stage: dict[str, float] = {}
    retrieve_s = mean_of("retrieve_latency_s")
    if retrieve_s is not None:
        stage["retrieve_s"] = retrieve_s
    rerank_s = mean_of("rerank_latency_s")
    if rerank_s is not None:
        stage["rerank_s"] = rerank_s
    if stage:
        generate = [float(row["latency_s"]) for row in case_rows if row.get("latency_s")]
        stage["generate_s"] = round(sum(generate) / len(generate), 4) if generate else 0.0
    return stage


def _collect_optional_telemetry(
    config: RunConfig, launcher: BackendLauncher
) -> TelemetryReport | None:
    if not config.measure_telemetry:
        return None
    from llb.backends.telemetry import collect_telemetry, nvidia_smi_power_reader

    return collect_telemetry(
        launcher,
        requested_context=config.max_model_len,
        timeout=config.request_timeout_s,
        vram_reader=_vram_reader(),
        power_reader=nvidia_smi_power_reader(),
    )
