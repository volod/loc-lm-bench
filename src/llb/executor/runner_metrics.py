"""Metrics for the runner: aggregate one ranked leaderboard row from the scored cases, attach the
answer-side + stage-latency signals, and collect optional backend telemetry.

`run_eval` calls `_aggregate` (rows + metrics) and `_collect_optional_telemetry`; the NVML VRAM
reader lives with the backend readers in `runner_backend.py`.
"""

from dataclasses import dataclass
from collections.abc import Mapping

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.core.contracts.results import CaseScoreRow, LeaderboardRow
from llb.core.contracts.runs import RunMetrics
from llb.core.contracts.hardware import TelemetryReport
from llb.eval import common as eval_common
from llb.executor.runner_backend import _vram_reader
from llb.scoring.leaderboard import ModelResult, rank_results
from llb.scoring.judge.model import judge_is_trusted


def _throughput(case_rows: list[CaseScoreRow], telemetry: Mapping[str, object]) -> float:
    """The rate the run is credited with: the steady measured rate, else what the cases observed."""
    steady_rate = telemetry.get("steady_tokens_per_s")
    if isinstance(steady_rate, int | float) and steady_rate > 0:
        return float(steady_rate)
    rates = [
        row["tokens_per_s"]
        for row in case_rows
        if row["status"] == eval_common.OK and row["tokens_per_s"] > 0
    ]
    return sum(rates) / len(rates) if rates else 0.0


def _mean_completion_tokens(case_rows: list[CaseScoreRow]) -> float:
    """Mean answer length over the cases that actually generated one."""
    lengths = [
        float(row["completion_tokens"])
        for row in case_rows
        if row["status"] == eval_common.OK and row["completion_tokens"] > 0
    ]
    return sum(lengths) / len(lengths) if lengths else 0.0


@dataclass(frozen=True, slots=True)
class _CaseMeans:
    """The per-case means a run publishes, read once so the result and the metrics cannot differ."""

    objective: float
    ranking: float
    token_precision: float
    token_recall: float
    found_rate: float
    mean_completion_tokens: float
    reliability: float
    tokens_per_s: float

    @classmethod
    def of(cls, case_rows: list[CaseScoreRow], telemetry: Mapping[str, object]) -> "_CaseMeans":
        """Average the scored cases, crediting the run with the throughput it actually reached."""
        n = len(case_rows)
        ok = [row for row in case_rows if row["status"] == eval_common.OK]
        return cls(
            objective=_mean(case_rows, "objective_score"),
            ranking=_mean(case_rows, "ranking_score"),
            token_precision=_mean(case_rows, "token_precision"),
            token_recall=_mean(case_rows, "token_recall"),
            found_rate=_mean(case_rows, "contains"),
            mean_completion_tokens=_mean_completion_tokens(case_rows),
            reliability=len(ok) / n if n else 0.0,
            tokens_per_s=_throughput(case_rows, telemetry),
        )


def _model_result(
    config: RunConfig,
    case_rows: list[CaseScoreRow],
    means: _CaseMeans,
    *,
    peak_vram: object,
    judge_score: float | None,
) -> ModelResult:
    """The one-model result the leaderboard ranks, built from the scored cases."""
    return ModelResult(
        model=config.model,
        backend=config.backend,
        objective_score=means.objective,
        n_cases=len(case_rows),
        reliability=means.reliability,
        tokens_per_s=means.tokens_per_s,
        peak_vram_mb=float(peak_vram) if isinstance(peak_vram, int | float) else None,
        judge_score=judge_score,
        ranking_score=means.ranking,
        token_precision=means.token_precision,
        token_recall=means.token_recall,
        found_rate=means.found_rate,
        mean_completion_tokens=means.mean_completion_tokens,
        case_objectives=[float(row["objective_score"]) for row in case_rows],
        case_ranking=[float(row["ranking_score"]) for row in case_rows],
        feasible=True,
    )


def _attach_power_metrics(
    metrics: RunMetrics, telemetry: Mapping[str, object], means: _CaseMeans
) -> None:
    """Power-derived efficiency rows, only on a run whose host actually measured power."""
    mean_power = telemetry.get("mean_power_w")
    if not isinstance(mean_power, int | float) or mean_power <= 0:
        return
    watts = float(mean_power)
    metrics["mean_power_w"] = round(watts, 2)
    metrics["tokens_per_watt"] = round(means.tokens_per_s / watts, 4)
    metrics["quality_per_watt"] = round(means.ranking * means.tokens_per_s / watts, 4)


def _aggregate(
    config: RunConfig,
    case_rows: list[CaseScoreRow],
    judge_rho: float | None,
    telemetry: Mapping[str, object],
    judge_score: float | None = None,
) -> tuple[list[LeaderboardRow], RunMetrics]:
    """Score the run: one ranked leaderboard row, and the metrics the manifest publishes."""
    means = _CaseMeans.of(case_rows, telemetry)
    result = _model_result(
        config,
        case_rows,
        means,
        peak_vram=telemetry.get("peak_vram_mb"),
        judge_score=judge_score,
    )
    # The judge is trusted only when calibrated AND it actually produced a score this run.
    trusted = judge_is_trusted(judge_rho, config.judge_threshold) and judge_score is not None
    rows = rank_results([result], judge_trusted=trusted)
    metrics: RunMetrics = {
        "objective_score": means.objective,
        "ranking_score": means.ranking,
        "token_precision": means.token_precision,
        "token_recall": means.token_recall,
        "found_rate": means.found_rate,
        "mean_completion_tokens": means.mean_completion_tokens,
        "reliability": means.reliability,
        "tokens_per_s": means.tokens_per_s,
    }
    _attach_power_metrics(metrics, telemetry, means)
    stage = _stage_latency(case_rows)
    if stage:
        metrics["stage_latency"] = stage
    if judge_score is not None:
        metrics["judge_score"] = round(judge_score, 4)
    _attach_guard_metrics(metrics, case_rows)
    _attach_answer_side_metrics(metrics, case_rows)
    _attach_envelope_metrics(metrics, case_rows)
    return rows, metrics


def _mean(case_rows: list[CaseScoreRow], key: str) -> float:
    values = [float(row[key]) for row in case_rows]  # type: ignore[literal-required]
    return sum(values) / len(values) if values else 0.0


def _attach_guard_metrics(metrics: RunMetrics, case_rows: list[CaseScoreRow]) -> None:
    """Run-level rates for the response-integrity guard (`llb.scoring.answer_guard`).

    The denominator is every scored case, exactly as `reliability`'s is, so the three read
    together: a model can be perfectly reliable by status and still deliver a third of its answers
    as English deliberation. `mean_reasoning_leak_chars` is what makes the throughput reading
    honest -- it prices the part of `mean_completion_tokens` that was never an answer.
    """
    rows = [row for row in case_rows if "reasoning_leak" in row]
    if not rows:
        return
    n = len(rows)
    metrics["reasoning_leak_rate"] = round(sum(1 for row in rows if row["reasoning_leak"]) / n, 4)
    metrics["language_mismatch_rate"] = round(
        sum(1 for row in rows if row.get("language_mismatch")) / n, 4
    )
    metrics["mean_reasoning_leak_chars"] = round(
        sum(int(row.get("reasoning_leak_chars", 0)) for row in rows) / n, 2
    )


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


def _attach_envelope_metrics(metrics: RunMetrics, case_rows: list[CaseScoreRow]) -> None:
    """Declared-answer-contract rates for the run (typed-rag-answer-envelope), when it ran.

    Conformance is the share of cases whose completion satisfied the contract; the two failure
    rates are kept apart because "did not emit JSON" and "emitted JSON of the wrong shape" call for
    different fixes. `envelope_repair_rate` is the share where the bounded reprompt was spent, so
    first-attempt conformance reads as `1 - envelope_repair_rate` and the repair's contribution as
    the gap between the two -- a formatting gain, never a reasoning one.
    """
    statuses = [str(row["envelope_status"]) for row in case_rows if "envelope_status" in row]
    if not statuses:
        return
    n = len(statuses)
    metrics["envelope_conformance"] = round(
        sum(1 for status in statuses if status == eval_common.OK) / n, 4
    )
    metrics["envelope_schema_invalid_rate"] = round(
        sum(1 for status in statuses if status == eval_common.SCHEMA_INVALID) / n, 4
    )
    metrics["envelope_malformed_rate"] = round(
        sum(1 for status in statuses if status == eval_common.MALFORMED) / n, 4
    )
    metrics["envelope_repair_rate"] = round(
        sum(1 for row in case_rows if row.get("repaired")) / n, 4
    )
    metrics["mean_claims"] = round(sum(int(row.get("n_claims", 0)) for row in case_rows) / n, 4)


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
    from llb.backends.telemetry import collect_telemetry
    from llb.backends.telemetry_samplers import nvidia_smi_power_reader

    return collect_telemetry(
        launcher,
        requested_context=config.max_model_len,
        timeout=config.request_timeout_s,
        vram_reader=_vram_reader(),
        power_reader=nvidia_smi_power_reader(),
    )
