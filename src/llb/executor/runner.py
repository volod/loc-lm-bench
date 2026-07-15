"""Minimal sequential eval runner -- the RAG core walking skeleton.

Orchestrates one (model, config) end to end: load eval items -> retrieve+generate per
case through the LangGraph RAG flow -> score objective correctness + collect retrieval
hits -> aggregate one ranked row -> persist the canonical manifest+scores (then mirror).

Every heavy collaborator is injectable (`store`, `launcher`, `runner_fn`, `mirror`), so
the whole vertical runs end to end in a unit test with fakes -- no FAISS, langgraph,
Ollama, or GPU. The default path wires the real components and uses the compiled
LangGraph app.

The run's building blocks live in sibling modules -- inputs (`runner_setup`), backend
lifecycle (`runner_backend`), judge scoring (`runner_judge`), metrics (`runner_metrics`),
and run-target resolution (`runner_target`). Import each helper from the module it lives in;
this module owns only `run_eval`.
"""

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.core.contracts.hardware import BackendMetadata
from llb.core.contracts.runs import EvalResult
from llb.eval import graph as eval_graph
from llb.executor import durability, durability_journal
from llb.executor.cases import batch_retrieval_records
from llb.executor.reporting import emit_summary
from llb.executor.runner_backend import (
    _preserve_failed_staging,
    _resolve_eval_runner,
)
from llb.executor.runner_judge import (
    JudgeScorer,
    _build_judge_metadata,
    _judge_cases,
    _write_calibration_worksheet,
)
from llb.executor.runner_metrics import (
    _aggregate,
    _collect_optional_telemetry,
)
from llb.executor.runner_setup import (
    _maybe_run_probes,
    _score_options,
    _select_eval_items,
)
from llb.executor.runner_target import (
    _eval_config_payload,
    _resolve_run_target,
)
from llb.goldset.schema import GoldItem
from llb.rag import retrieval
from llb.scoring.leaderboard import format_table
from llb.tracking.manifest import RunManifest, persist_run

RagState = eval_graph.RagState
_LOG = logging.getLogger(__name__)


def run_eval(
    config: RunConfig,
    *,
    items: list[GoldItem] | None = None,
    store: Any = None,
    launcher: BackendLauncher | None = None,
    runner_fn: Callable[[GoldItem], RagState] | None = None,
    prompt_package: Any | None = None,
    prompt_system_provenance: Mapping[str, object] | None = None,
    mirror: Callable[[RunManifest, Path], None] | None = None,
    judge_rho: float | None = None,
    judge_scorer: JudgeScorer | None = None,
    limit: int | None = None,
    split: str = "final",
    worksheet: Path | str | None = None,
    evict: bool = False,
    wait: bool = False,
    emit: bool = True,
    resume: Path | str | None = None,
    max_case_retries: int = 2,
    retry_backoff_s: float = 1.0,
    max_backend_relaunches: int = 1,
    sleep: Callable[[float], None] | None = None,
) -> EvalResult:
    """Run the skeleton and return {rows, metrics, paths, table}.

    `worksheet` (a path) emits a judge-calibration worksheet pre-filled with this run's
    model answers (the human only adds ratings); pair it with `split="calibration"`.

    The run is durable (the durable-eval-runner): completed cases journal to
    `cases.progress.jsonl` in the staging dir, transient per-case transport failures retry
    (`max_case_retries` / `retry_backoff_s`), and a crashed launcher-owned backend relaunches up to
    `max_backend_relaunches` times. `resume=<run-dir>` continues an interrupted run from its journal
    instead of re-spending model calls; the config fingerprint and goldset digest must match.
    """
    items = _select_eval_items(config, items, split, limit)
    if not items:
        raise SystemExit(
            f"no verified '{split}' items in {config.goldset_path} "
            "(only items with verified=true are scored; public-reused sets ship "
            "verified=false pending human review)"
        )
    config_payload = _eval_config_payload(config, items, prompt_system_provenance)
    run_timestamp, run_id, run_dir, staging_dir = _resolve_run_target(
        config, resume, config_payload, items, split
    )

    active_launcher: BackendLauncher | None = None
    counters = durability_journal.DurabilityCounters()
    try:
        active_launcher, runner_fn, store, contention = _resolve_eval_runner(
            config,
            store=store,
            launcher=launcher,
            runner_fn=runner_fn,
            prompt_package=prompt_package,
            staging_dir=staging_dir,
            evict=evict,
            wait=wait,
        )
        embedder = (
            store.embedder if (config.score_semantic and hasattr(store, "embedder")) else None
        )
        score_options = _score_options(config)
        policy = durability_journal.RetryPolicy(
            max_case_retries=max_case_retries,
            retry_backoff_s=retry_backoff_s,
            max_backend_relaunches=max_backend_relaunches,
        )
        with active_launcher as backend:

            def relaunch() -> None:
                backend.stop()
                backend.start()

            batch, counters = durability.execute_cases_durable(
                items,
                runner_fn,
                embedder,
                journal=durability_journal.CaseJournal(
                    durability_journal.journal_path(staging_dir)
                ),
                policy=policy,
                relaunch=relaunch,
                sleep=sleep if sleep is not None else time.sleep,
                counters=counters,
                options=score_options,
            )
            telemetry_report = _collect_optional_telemetry(config, backend)
            probe_report = _maybe_run_probes(config, items, store, backend)
    except KeyboardInterrupt:
        _preserve_failed_staging(
            active_launcher, config, resume, run_dir, staging_dir, interrupted=True
        )
        raise
    except BaseException:
        _preserve_failed_staging(
            active_launcher, config, resume, run_dir, staging_dir, interrupted=False
        )
        raise

    backend_telemetry: BackendMetadata = (
        active_launcher.telemetry() if hasattr(active_launcher, "telemetry") else {}
    )
    effective_telemetry = {**backend_telemetry, **(telemetry_report or {})}
    judge_score = _judge_cases(config, batch, judge_rho, judge_scorer)
    rows, metrics = _aggregate(config, batch.rows, judge_rho, effective_telemetry, judge_score)
    if probe_report is not None:
        metrics["abstention_accuracy"] = round(probe_report.abstention_accuracy, 4)
        metrics["n_probes"] = probe_report.n_probes
    retrieval_metrics = retrieval.evaluate_retrieval(batch.retrieval_pairs, config.top_k)

    manifest = RunManifest(
        run_id=run_id,
        run_name=config.run_name,
        split=split,
        config=config_payload,
        metrics=metrics,
        retrieval=retrieval_metrics,
        judge=_build_judge_metadata(config, judge_rho),
        telemetry=telemetry_report,
        contention=contention,
        durability=counters.as_status(),
        prompt_system_provenance=dict(prompt_system_provenance)
        if prompt_system_provenance is not None
        else None,
        n_cases=len(batch.rows),
    )
    durability_journal.drop_journal(staging_dir)
    paths = persist_run(
        manifest,
        batch.rows,
        run_dir,
        mirror=mirror,
        staging_dir=staging_dir,
        retrieval_rows=batch_retrieval_records(batch),
    )

    worksheet_rows = 0
    if worksheet is not None:
        worksheet_rows = _write_calibration_worksheet(config, batch, worksheet, judge_scorer)
        paths["worksheet"] = str(worksheet)

    if probe_report is not None:
        from llb.eval.insufficient_context import write_probe

        paths.update(write_probe(probe_report, run_dir))  # type: ignore[typeddict-item]

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
            metrics,
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
