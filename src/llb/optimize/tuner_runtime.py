"""Focused tuner runtime: default evaluate hooks, the study's served-window resolution, and
MLflow trial logging.

Store caching lives in ``llb.optimize.store_registry`` (re-exported here for callers).
"""

import logging
import time
from typing import Any, Callable

from llb.core.config import RunConfig
from llb.core.contracts.runs import EvalResult
from llb.optimize.objectives import TrialMetrics
from llb.optimize.store_registry import (
    StoreRegistry,
    chunking_fingerprint,
    store_fingerprint,
    study_stores_dir,
)
from llb.core.contracts.models import ModelSpec
from llb.backends.context_fit import bound_max_context
from llb.optimize.tuning_space import (
    FINAL_SPLIT,
    TUNING_SPLIT,
)

_LOG = logging.getLogger(__name__)

# Back-compat aliases used by older tests / imports.
_store_fingerprint = store_fingerprint
_chunking_fingerprint = chunking_fingerprint


def resolve_study_window(
    base_config: RunConfig,
    *,
    model_spec: ModelSpec | None,
    vram_mib: int,
    ram_mib: int,
    served_max_model_len: int | None = None,
    probe: bool = True,
) -> tuple[int | None, dict[str, object]]:
    """The served window a whole study prunes against, plus the provenance to record.

    Resolved ONCE, not per trial: what the backend serves does not depend on the RAG parameters a
    trial samples, while the declared side does -- `tune_context_budget` samples a `context_budget`
    that tightens the declared window per trial, so only the served side is a study constant. The
    reported binding is therefore the study's own (from `base_config`); a trial that samples a
    smaller budget is bound tighter still, never looser.

    `probe=False` (or an explicit `served_max_model_len`) is the injected-probe seam CI runs on.
    """
    served = served_max_model_len
    if served is None and probe:
        from llb.backends.served_window import probe_served_window

        served = probe_served_window(base_config)
    window, source = bound_max_context(base_config, model_spec, vram_mib, ram_mib, served)
    declared, _ = bound_max_context(base_config, model_spec, vram_mib, ram_mib)
    provenance: dict[str, object] = {
        "declared_max_model_len": declared or None,
        "served_max_model_len": served,
        "budget_source": source,
    }
    _LOG.info(
        "[tune] prune window: %s tok (%s; declared %s, served %s)",
        window or "unbounded",
        source,
        declared or "unbounded",
        served if served is not None else "unprobed",
    )
    return served, provenance


TrialCallback = Callable[[dict[str, Any]], None]  # per-completed-trial hook (e.g. MLflow child)

__all__ = [
    "StoreRegistry",
    "TrialCallback",
    "resolve_study_window",
    "_LOG",
    "_build_store",
    "_chunking_fingerprint",
    "_run_eval_final",
    "_run_eval_metrics",
    "_run_eval_quality",
    "_store_fingerprint",
    "mlflow_trial_logger",
    "study_stores_dir",
]


def _build_store(config: RunConfig) -> Any:
    """Build a store and apply the trial's fusion + rerank knobs (single-eval path)."""
    from llb.optimize.store_registry import _apply_query_knobs, _build_bare_store

    return _apply_query_knobs(_build_bare_store(config), config)


def _frontier_cost_usd(result: EvalResult) -> float:
    """Read frontier spend from the run manifest judge budget block when present."""
    manifest: dict[str, Any] = result.get("manifest") or {}  # type: ignore[assignment]
    if not isinstance(manifest, dict):
        return 0.0
    judge = manifest.get("judge")
    if not isinstance(judge, dict):
        return 0.0
    budget = judge.get("budget")
    if not isinstance(budget, dict):
        return 0.0
    return float(budget.get("cost_usd") or 0.0)


def _run_eval_quality(config: RunConfig) -> tuple[float, float]:
    """Default stage-1 objective: build the config's store, score the tuning split, and return
    (quality, throughput) so the tuner can tie-break equal-quality configs by speed."""
    from llb.executor.runner import run_eval

    result = run_eval(config, store=_build_store(config), split=TUNING_SPLIT, emit=False)
    rows = result["rows"]
    if not rows:
        return 0.0, 0.0
    return float(rows[0]["quality"]), float(rows[0].get("tokens_per_s", 0.0))


def _run_eval_metrics(
    config: RunConfig,
    *,
    limit: int | None = None,
    stores: StoreRegistry | None = None,
) -> TrialMetrics:
    """Multi-objective evaluate: quality + generate latency + optional frontier cost."""
    from llb.executor.runner import run_eval

    store = stores.get(config) if stores is not None else _build_store(config)
    started = time.perf_counter()
    result = run_eval(config, store=store, split=TUNING_SPLIT, emit=False, limit=limit)
    wall_s = time.perf_counter() - started
    rows = result["rows"]
    if not rows:
        return TrialMetrics(quality=0.0, latency_s=wall_s)
    metrics = result.get("metrics") or {}
    stage = metrics.get("stage_latency") if isinstance(metrics, dict) else None
    # Prefer mean generate latency when measured -- tracks context size better than wall-clock.
    generate_s = stage.get("generate_s") if isinstance(stage, dict) else None
    latency_s = (
        float(generate_s) if isinstance(generate_s, int | float) and generate_s > 0 else wall_s
    )
    return TrialMetrics(
        quality=float(rows[0]["quality"]),
        latency_s=latency_s,
        cost_usd=_frontier_cost_usd(result),
        throughput=float(rows[0].get("tokens_per_s", 0.0)),
    )


def _run_eval_final(config: RunConfig, *, limit: int | None = None) -> EvalResult:
    """Default stage-2 run on the final split; ``limit`` is for explicit smoke/evidence runs."""
    from llb.executor.runner import run_eval

    return run_eval(config, store=_build_store(config), split=FINAL_SPLIT, emit=True, limit=limit)


def mlflow_trial_logger(study_name: str) -> TrialCallback:
    """A best-effort `on_trial` hook that mirrors each Optuna trial as a NESTED MLflow run under
    a `<study_name>` parent, so the stage-1 search is inspectable alongside the stage-2 entry.
    Any MLflow error is swallowed (tuning never fails because tracking is unavailable)."""

    def log(record: dict[str, Any]) -> None:
        try:
            import mlflow

            if mlflow.active_run() is None:
                mlflow.start_run(run_name=f"{study_name}-search")
            with mlflow.start_run(run_name=f"trial-{record['number']}", nested=True):
                mlflow.log_metric("quality", float(record.get("quality", 0.0)))
                mlflow.log_metric("throughput", float(record.get("throughput", 0.0)))
                if "latency_s" in record:
                    mlflow.log_metric("latency_s", float(record["latency_s"]))
                if "cost_usd" in record:
                    mlflow.log_metric("cost_usd", float(record["cost_usd"]))
                skip = {"quality", "throughput", "latency_s", "cost_usd"}
                params = {k: v for k, v in record.items() if k not in skip}
                mlflow.log_params(params)
        except Exception:  # pragma: no cover - tracking is best-effort
            _LOG.debug("[tune] MLflow trial logging skipped for trial %s", record.get("number"))

    return log
