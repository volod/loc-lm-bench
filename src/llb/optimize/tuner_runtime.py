"""Focused tuner runtime implementation."""

import logging
from typing import Any, Callable
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult
from llb.optimize.tuning_space import (
    FINAL_SPLIT,
    TUNING_SPLIT,
)

_LOG = logging.getLogger(__name__)

TrialCallback = Callable[[dict[str, Any]], None]  # per-completed-trial hook (e.g. MLflow child)


def _build_store(config: RunConfig) -> Any:
    from llb.rag.rerank import maybe_wrap_reranker
    from llb.rag.store import RagStore

    store = RagStore.build(
        config.corpus_root,
        config.strategy,
        config.chunk_size,
        config.chunk_overlap,
        config.embedding_model,
        mode=config.retrieval_mode,
        child_size=config.child_chunk_size,
        lexical_lemmas=config.lexical_lemmas,
    )
    # The store is injected into run_eval directly (no _load_store pass), so the trial's
    # fusion + rerank knobs must be applied here to take effect.
    store.fusion_weight = config.fusion_weight
    store.fusion_candidates = config.fusion_candidates
    return maybe_wrap_reranker(store, config)


def _run_eval_quality(config: RunConfig) -> tuple[float, float]:
    """Default stage-1 objective: build the config's store, score the tuning split, and return
    (quality, throughput) so the tuner can tie-break equal-quality configs by speed."""
    from llb.executor.runner import run_eval

    result = run_eval(config, store=_build_store(config), split=TUNING_SPLIT, emit=False)
    rows = result["rows"]
    if not rows:
        return 0.0, 0.0
    return float(rows[0]["quality"]), float(rows[0].get("tokens_per_s", 0.0))


def _run_eval_final(config: RunConfig) -> EvalResult:
    """Default stage-2 run: score the winning config on the full final split (the entry)."""
    from llb.executor.runner import run_eval

    return run_eval(config, store=_build_store(config), split=FINAL_SPLIT, emit=True)


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
                params = {k: v for k, v in record.items() if k not in ("quality", "throughput")}
                mlflow.log_params(params)
        except Exception:  # pragma: no cover - tracking is best-effort
            _LOG.debug("[tune] MLflow trial logging skipped for trial %s", record.get("number"))

    return log
