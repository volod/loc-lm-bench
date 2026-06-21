"""Minimal sequential eval runner -- the Milestone 1 walking skeleton.

Orchestrates one (model, config) end to end: load eval items -> retrieve+generate per
case through the LangGraph RAG flow -> score objective correctness + collect retrieval
hits -> aggregate one ranked row -> persist the canonical manifest+scores (then mirror).

Every heavy collaborator is injectable (`store`, `launcher`, `runner_fn`, `mirror`), so
the whole vertical runs end to end in a unit test with fakes -- no FAISS, langgraph,
Ollama, or GPU. The default path wires the real components and uses the compiled
LangGraph app.
"""

import logging
import shutil
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.backends.base import BackendLauncher
from llb.config import RunConfig
from llb.contracts import (
    BackendMetadata,
    CaseScoreRow,
    EvalResult,
    JudgeInputRecord,
    JudgeScore,
    JudgeStatus,
    LeaderboardRow,
    RunMetrics,
    TelemetryReport,
)
from llb.eval import graph as eval_graph
from llb.executor.cases import CaseBatch, execute_cases, spans_as_dicts
from llb.executor.reporting import emit_summary
from llb.goldset.schema import GoldItem, load_goldset
from llb.rag import retrieval
from llb.scoring.aggregate import ModelResult, format_table, rank_results
from llb.scoring.judge import judge_is_trusted, run_judge
from llb.tracking.manifest import RunManifest, persist_run

JudgeScorer = Callable[[list[JudgeInputRecord], str], list[JudgeScore]]

RagState = eval_graph.RagState
_RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"
_LOG = logging.getLogger(__name__)


def _preserve_backend_log(launcher: BackendLauncher, config: RunConfig) -> None:
    """Copy a failed backend's startup log out of the staging dir (which is about to be
    removed) into the persistent logs dir, so a launch failure stays diagnosable instead of
    vanishing with the staging bundle (e.g. a vLLM engine that dies during startup)."""
    log_path = getattr(launcher, "log_path", None)
    src = Path(log_path) if log_path else None
    if src is None or not src.exists():
        return
    dest_dir = config.data_dir / "llb" / "logs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"failed-{src.stem}-{stamp}.log"
    try:
        shutil.copyfile(src, dest)
    except OSError:
        return
    _LOG.error("[run-eval] backend failed to start; startup log preserved -> %s", dest)


def _load_eval_items(config: RunConfig, split: str, limit: int | None) -> list[GoldItem]:
    if not config.goldset_path.exists():
        raise SystemExit(
            f"gold set not found: {config.goldset_path}\n"
            "  use the committed fixture with --goldset "
            "samples/goldsets/ua_squad_postedited_v1/goldset.jsonl,\n"
            "  or create unverified development material with `make ingest-uk-squad`."
        )
    items = [
        item for item in load_goldset(config.goldset_path) if item.split == split and item.verified
    ]
    items.sort(key=lambda it: it.id)
    return items[:limit] if limit is not None else items


def _select_eval_items(
    config: RunConfig,
    items: list[GoldItem] | None,
    split: str,
    limit: int | None,
) -> list[GoldItem]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if items is None:
        return _load_eval_items(config, split, limit)
    selected = sorted(
        (item for item in items if item.split == split and item.verified),
        key=lambda item: item.id,
    )
    return selected[:limit] if limit is not None else selected


def _make_launcher(config: RunConfig, log_dir: Path | None = None) -> BackendLauncher:
    if config.backend == "ollama":
        from llb.backends.ollama import OllamaLauncher

        return OllamaLauncher(config.model, host=config.ollama_host)
    if config.backend == "vllm":
        from llb.backends.vllm import VllmLauncher

        return VllmLauncher(
            config.model,
            host=config.vllm_host,
            port=config.vllm_port,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
            quantization=config.quantization,
            log_dir=log_dir,
        )
    raise SystemExit(f"backend '{config.backend}' is not wired yet (Ollama + vLLM supported).")


def _vram_reader() -> Callable[[], int] | None:
    """Best-effort NVML reader for telemetry (None when the [telemetry] extra/GPU is absent)."""
    try:
        from llb.executor.vram import nvml_reader

        return nvml_reader()
    except (Exception, SystemExit):  # nvml_reader raises SystemExit when [telemetry] is absent
        return None


def _default_runner_fn(
    config: RunConfig, store: Any, launcher: BackendLauncher
) -> Callable[[GoldItem], RagState]:
    app = eval_graph.build_rag_graph(
        store,
        launcher,
        config.top_k,
        config.max_tokens,
        config.temperature,
        config.request_timeout_s,
    )

    def run(item: GoldItem) -> RagState:
        return eval_graph.run_case(app, item.question, spans_as_dicts(item))

    return run


def _judge_records(batch: CaseBatch) -> list[JudgeInputRecord]:
    """The (question, answer, retrieved-contexts) record per case the judge scores."""
    return [
        {
            "question": item.question,
            "answer": answer,
            "contexts": [str(chunk.get("text", "")) for chunk in retrieved],
        }
        for (item, answer), (retrieved, _spans) in zip(batch.answers, batch.retrieval_pairs)
    ]


def _judge_value(score: JudgeScore) -> float:
    """One scalar judge rating per case: the mean of faithfulness + answer-relevancy."""
    return (score["faithfulness"] + score["answer_relevancy"]) / 2.0


def _configured_judge_scorer(config: RunConfig, scorer: JudgeScorer | None) -> JudgeScorer:
    """Bind the configured endpoint while preserving the injectable scorer seam."""
    if scorer is not None:
        return scorer
    from llb.scoring.judge import deepeval_scorer

    def score(records: list[JudgeInputRecord], model: str) -> list[JudgeScore]:
        return deepeval_scorer(records, model, base_url=config.judge_base_url)

    return score


def _judge_cases(
    config: RunConfig,
    batch: CaseBatch,
    judge_rho: float | None,
    scorer: JudgeScorer | None,
) -> float | None:
    """Score answers with the GATED judge (Premise 2) and attach per-case judge scores.

    Returns the mean per-case judge score ONLY when the judge is configured AND trusted
    (calibration rho >= threshold); otherwise the judge stays a demoted diagnostic and objective
    correctness ranks alone. The per-case judge value is the mean of faithfulness + relevancy.
    """
    if config.judge_model is None:
        return None
    outcome = run_judge(
        _judge_records(batch),
        config.judge_model,
        judge_rho,
        config.judge_threshold,
        scorer=_configured_judge_scorer(config, scorer),
    )
    if not outcome.trusted or not outcome.scores:
        _LOG.info("[run-eval] judge demoted (%s); objective ranks alone", outcome.reason)
        return None
    per_case = [_judge_value(s) for s in outcome.scores]
    for row, value in zip(batch.rows, per_case):
        row["judge_score"] = round(value, 4)
    return sum(per_case) / len(per_case) if per_case else None


def _judge_ratings(
    config: RunConfig, batch: CaseBatch, scorer: JudgeScorer | None
) -> list[float] | None:
    """Run the judge UNGATED and return one rating per case (M3.8 calibration scaffolding).

    Calibration measures whether the judge AGREES with humans, so the judge runs regardless of
    its (not-yet-known) trust -- the gate is irrelevant here. Returns None when no judge is
    configured; raises if the judge backend is unavailable (so the worksheet path can warn).
    """
    if config.judge_model is None:
        return None
    score_fn = _configured_judge_scorer(config, scorer)
    scores = score_fn(_judge_records(batch), config.judge_model)
    return [_judge_value(s) for s in scores]


def _aggregate(
    config: RunConfig,
    case_rows: list[CaseScoreRow],
    judge_rho: float | None,
    telemetry: Mapping[str, object],
    judge_score: float | None = None,
) -> tuple[list[LeaderboardRow], RunMetrics]:
    n = len(case_rows)
    objective = sum(r["objective_score"] for r in case_rows) / n if n else 0.0
    ok = [r for r in case_rows if r["status"] == eval_graph.OK]
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
    if judge_score is not None:
        metrics["judge_score"] = round(judge_score, 4)
    return rows, metrics


def _run_timestamp(run_id: str) -> str:
    now = datetime.now(timezone.utc).strftime(_RUN_TIMESTAMP_FORMAT)
    return f"{now}-{run_id}"


def _collect_optional_telemetry(
    config: RunConfig, launcher: BackendLauncher
) -> TelemetryReport | None:
    if not config.measure_telemetry:
        return None
    from llb.backends.telemetry import collect_telemetry

    return collect_telemetry(
        launcher,
        requested_context=config.max_model_len,
        timeout=config.request_timeout_s,
        vram_reader=_vram_reader(),
    )


def run_eval(
    config: RunConfig,
    *,
    items: list[GoldItem] | None = None,
    store: Any = None,
    launcher: BackendLauncher | None = None,
    runner_fn: Callable[[GoldItem], RagState] | None = None,
    mirror: Callable[[RunManifest, Path], None] | None = None,
    judge_rho: float | None = None,
    judge_scorer: JudgeScorer | None = None,
    limit: int | None = None,
    split: str = "final",
    worksheet: Path | str | None = None,
    emit: bool = True,
) -> EvalResult:
    """Run the skeleton and return {rows, metrics, paths, table}.

    `worksheet` (a path) emits a judge-calibration worksheet pre-filled with this run's
    model answers (the human only adds ratings); pair it with `split="calibration"`.
    """
    items = _select_eval_items(config, items, split, limit)
    if not items:
        raise SystemExit(
            f"no verified '{split}' items in {config.goldset_path} "
            "(only items with verified=true are scored; public-reused sets ship "
            "verified=false pending human review)"
        )

    run_id = uuid.uuid4().hex[:12]
    run_timestamp = _run_timestamp(run_id)
    run_dir = config.run_dir(run_timestamp)
    staging_dir = config.run_staging_dir(run_timestamp)
    if launcher is None:
        launcher = _make_launcher(config, log_dir=staging_dir / "vllm")
    if runner_fn is None:
        if store is None:
            from llb.rag.store import RagStore

            store = RagStore.load(config.index_dir())
        runner_fn = _default_runner_fn(config, store, launcher)

    embedder = store.embedder if (config.score_semantic and hasattr(store, "embedder")) else None

    try:
        with launcher:
            batch = execute_cases(items, runner_fn, embedder)
            telemetry_report = _collect_optional_telemetry(config, launcher)
    except BaseException:
        _preserve_backend_log(launcher, config)
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    backend_telemetry: BackendMetadata = (
        launcher.telemetry() if hasattr(launcher, "telemetry") else {}
    )
    effective_telemetry = {**backend_telemetry, **(telemetry_report or {})}
    judge_score = _judge_cases(config, batch, judge_rho, judge_scorer)
    rows, metrics = _aggregate(config, batch.rows, judge_rho, effective_telemetry, judge_score)
    retrieval_metrics = retrieval.evaluate_retrieval(batch.retrieval_pairs, config.top_k)

    judge_metadata: JudgeStatus = {
        "calibration_rho": judge_rho,
        "threshold": config.judge_threshold,
        "trusted": judge_is_trusted(judge_rho, config.judge_threshold),
    }
    if config.judge_model is not None:
        from llb.scoring.judge import judge_experiment_metadata

        experiment_metadata = judge_experiment_metadata(config.judge_model, config.judge_base_url)
        judge_metadata["provider"] = experiment_metadata["provider"]
        judge_metadata["model"] = experiment_metadata["model"]
        judge_metadata["base_url"] = experiment_metadata["base_url"]
        judge_metadata["prompt_language"] = experiment_metadata["prompt_language"]
        judge_metadata["metrics"] = experiment_metadata["metrics"]
    manifest = RunManifest(
        run_id=run_id,
        run_name=config.run_name,
        split=split,
        config=config.fingerprint(),
        metrics=metrics,
        retrieval=retrieval_metrics,
        judge=judge_metadata,
        telemetry=telemetry_report,
        n_cases=len(batch.rows),
    )
    paths = persist_run(
        manifest,
        batch.rows,
        run_dir,
        mirror=mirror,
        staging_dir=staging_dir,
    )

    worksheet_rows = 0
    if worksheet is not None:
        from llb.judge.calibration import write_filled_worksheet

        # For a calibration worksheet, also run the judge UNGATED so the judge_rating column is
        # pre-filled and the human only adds human_rating; rho(human, judge) follows.
        judge_ratings: list[float] | None = None
        if config.judge_model is not None:
            try:
                judge_ratings = _judge_ratings(config, batch, judge_scorer)
            except (Exception, SystemExit) as exc:
                _LOG.warning(
                    "[run-eval] judge unavailable for the worksheet (%s); judge_rating left blank "
                    "-- pick the judge (OQ2) and install its backend to calibrate.",
                    exc,
                )
        worksheet_path = Path(worksheet)
        worksheet_rows = write_filled_worksheet(
            batch.answers, worksheet_path, judge_ratings=judge_ratings
        )
        paths["worksheet"] = str(worksheet)

    table = format_table(rows)
    if emit:
        emit_summary(
            config,
            len(batch.rows),
            retrieval_metrics,
            table,
            telemetry_report,
            paths,
            worksheet,
            worksheet_rows,
        )
    return {
        "rows": rows,
        "metrics": metrics,
        "retrieval": retrieval_metrics,
        "paths": paths,
        "table": table,
        "telemetry": telemetry_report,
        "manifest": manifest,
        "run_timestamp": run_timestamp,
    }
