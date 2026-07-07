"""Two-stage Optuna tuning of backend + RAG parameters (Optuna tuning).

The split discipline keeps the leaderboard honest: STAGE 1 searches the RAG/backend space on
the disjoint `tuning` split (a proxy -- never the final gold items), STAGE 2 scores ONLY the
winning config on the full `final` split, and only that stage-2 run is the leaderboard entry.
The embedding is PINNED (Premise 4) and is never a search dimension.

Search space (the chunking machinery already exists in RAG core):
  strategy   {fixed, sentence, recursive, markdown, semantic}
  chunk_size / chunk_overlap (as a fraction, so overlap < size always holds)
  top_k
  retrieval_mode {flat, parent_child} x child_chunk_size

Over-context configs are PRUNED before they are ever run: a big `top_k x chunk_size` retrieved
context can exceed what the model can hold, and that depends on the RAG params, not just the
model -- so the prune is a real search-space constraint, not a no-op. The Optuna study uses a
persistent SQLite backend so a killed sweep resumes (`load_if_exists`).

`optuna` is imported lazily (the `[track]` extra), so this module imports in the base install;
the search-space + fit helpers are pure and unit-testable, and the heavy per-trial evaluation
is injectable.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from llb.backends.planner import plan_model
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult, ModelSpec

_LOG = logging.getLogger(__name__)

TUNING_SPLIT = "tuning"
FINAL_SPLIT = "final"
OPTUNA_METHOD = "optuna"

STRATEGIES = ["fixed", "sentence", "recursive", "markdown", "semantic"]
RETRIEVAL_MODES = ["flat", "parent_child"]
CHARS_PER_TOKEN = 3.0  # UA measured ~0.33 tok/char in real-model validation -> ~3 chars/token
PROMPT_HEADROOM_TOKENS = 512  # system prompt + question + answer headroom

# config -> quality on the tuning split, OR (quality, throughput) for the latency tie-break.
Objective = Callable[[RunConfig], "float | tuple[float, float]"]
TrialCallback = Callable[[dict[str, Any]], None]  # per-completed-trial hook (e.g. MLflow child)


def with_isolation(
    evaluate: Objective,
    *,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
    gpu_sampler: Callable[[], list[Any]] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Objective:
    """Wrap a trial `evaluate` so each Optuna trial runs through the SAME `isolate_cell` contract
    as a sweep cell (isolation reclaim): VRAM baseline -> trial -> PID-attributed reclaim gate (a leaked trial
    aborts the study) -> capped thermal cooldown. This reuses the executor's cell isolation for
    tuning, so a trial that leaks VRAM cannot bias later trials' fit/throughput."""
    import functools

    from llb.executor.isolation import isolate_cell

    def run(config: RunConfig) -> "float | tuple[float, float]":
        out, _outcome = isolate_cell(
            functools.partial(evaluate, config),
            backend=config.backend,
            vram_reader=vram_reader,
            pid_usage_reader=pid_usage_reader,
            gpu_sampler=gpu_sampler,
            sleep=sleep,
        )
        return out

    return run


# Substrings that mark a measured out-of-memory / capacity failure -> prune, do not crash.
_OOM_MARKERS = ("out of memory", "outofmemory", "cuda error", "no available memory", "kv cache")


SERVING_MAX_MODEL_LEN = [4096, 8192, 16384]


def suggest_overrides(trial: Any, backend: str = "ollama") -> dict[str, Any]:
    """Sample one config from an Optuna trial (embedding is pinned, never sampled).

    RAG params are always sampled. BACKEND-AWARE serving knobs are sampled only when the
    resolved backend actually exposes them: `gpu_memory_utilization` / `max_model_len` are vLLM
    concepts, so sampling them for Ollama would tune dead parameters (llama.cpp knobs land with
    that launcher).
    """
    strategy = trial.suggest_categorical("strategy", STRATEGIES)
    chunk_size = trial.suggest_int("chunk_size", 256, 1280, step=64)
    overlap_frac = trial.suggest_float("overlap_frac", 0.0, 0.4)
    mode = trial.suggest_categorical("retrieval_mode", RETRIEVAL_MODES)
    top_k = trial.suggest_int("top_k", 3, 12)
    overrides: dict[str, Any] = {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": int(chunk_size * overlap_frac),
        "top_k": top_k,
        "retrieval_mode": mode,
    }
    if mode == "parent_child":
        # child must stay below chunk_size (and the validator wants overlap < child_size).
        ceiling = max(128, chunk_size - 64)
        child = trial.suggest_int("child_chunk_size", 128, 640, step=32)
        overrides["child_chunk_size"] = min(child, ceiling)
    if backend == "vllm":
        overrides["gpu_memory_utilization"] = trial.suggest_float(
            "gpu_memory_utilization", 0.70, 0.90, step=0.05
        )
        overrides["max_model_len"] = trial.suggest_categorical(
            "max_model_len", SERVING_MAX_MODEL_LEN
        )
    return overrides


def estimate_prompt_tokens(config: RunConfig) -> int:
    """Rough tokens consumed by the retrieved context + headroom + the requested completion."""
    retrieved_chars = config.top_k * config.chunk_size
    return int(retrieved_chars / CHARS_PER_TOKEN) + PROMPT_HEADROOM_TOKENS + config.max_tokens


def effective_max_context(
    config: RunConfig, model_spec: ModelSpec, vram_mib: int, ram_mib: int
) -> int:
    """The smallest of: the planner's max context for the host, the model window, and the
    served `max_model_len` cap. 0 means "cannot bound" (so the caller should not prune)."""
    row = plan_model(model_spec, vram_mib, ram_mib)
    ctx = row["ctx_max"] or int(model_spec.get("max_context") or 0)
    if config.max_model_len:
        ctx = min(ctx, config.max_model_len) if ctx else config.max_model_len
    return ctx


def fits_context(
    config: RunConfig, model_spec: ModelSpec | None, vram_mib: int, ram_mib: int
) -> bool:
    """True if the retrieved prompt fits the effective context. No spec -> cannot judge -> True."""
    if model_spec is None:
        return True
    ctx = effective_max_context(config, model_spec, vram_mib, ram_mib)
    return ctx <= 0 or estimate_prompt_tokens(config) <= ctx


@dataclass
class TuneResult:
    best_config: RunConfig
    best_value: float
    n_trials: int
    n_complete: int
    n_pruned: int
    study_name: str
    storage: str | None


@dataclass
class TwoStageResult:
    tune: TuneResult
    final: EvalResult  # the stage-2 run on the full final split -- the leaderboard entry


def _is_oom(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _OOM_MARKERS)


def make_objective(
    base_config: RunConfig,
    evaluate: Objective,
    *,
    model_spec: ModelSpec | None = None,
    vram_mib: int = 0,
    ram_mib: int = 0,
    on_trial: TrialCallback | None = None,
) -> Callable[[Any], float]:
    """Build the Optuna objective: sample -> validate -> prune over-context -> evaluate.

    The estimate-based context prune happens BEFORE a trial runs; a MEASURED OOM during the
    run (the model actually ran out of memory) prunes the trial too, so a too-aggressive
    serving config is dropped instead of crashing the whole study.
    """
    import optuna

    def objective(trial: Any) -> float:
        overrides = suggest_overrides(trial, backend=base_config.backend)
        try:
            config = base_config.with_overrides(**overrides)
        except ValueError as exc:  # e.g. overlap >= chunk_size after rounding
            raise optuna.TrialPruned(f"invalid config: {exc}") from None
        if not fits_context(config, model_spec, vram_mib, ram_mib):
            raise optuna.TrialPruned(
                f"retrieved context ~{estimate_prompt_tokens(config)} tok exceeds the model window"
            )
        trial.set_user_attr("overrides", overrides)
        try:
            outcome = evaluate(config)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            if _is_oom(exc):
                raise optuna.TrialPruned(f"measured OOM: {exc}") from None
            raise
        quality, throughput = outcome if isinstance(outcome, tuple) else (outcome, 0.0)
        trial.set_user_attr("throughput", throughput)
        if on_trial is not None:
            on_trial(
                {"number": trial.number, "quality": quality, "throughput": throughput, **overrides}
            )
        return quality

    return objective


def tune(
    base_config: RunConfig,
    *,
    n_trials: int,
    study_name: str,
    evaluate: Objective | None = None,
    storage: str | None = None,
    seed: int = 13,
    model_spec: ModelSpec | None = None,
    vram_mib: int = 0,
    ram_mib: int = 0,
    on_trial: TrialCallback | None = None,
    isolate: bool = False,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
    gpu_sampler: Callable[[], list[Any]] | None = None,
) -> TuneResult:
    """Stage 1: search the RAG/backend space on the tuning split; return the best config.

    `storage` defaults to a persistent SQLite study under ``$DATA_DIR/optuna/`` so a killed
    run resumes (set `storage=None` only in tests for an in-memory study). Ties on quality are
    broken by higher measured throughput, so the faster of two equal-quality configs wins. With
    `isolate=True`, each trial runs through `with_isolation` (the executor's per-cell VRAM/thermal
    gate), so a trial that leaks VRAM aborts the study instead of biasing later trials.
    """
    import optuna

    evaluate = evaluate or _run_eval_quality
    if isolate:
        evaluate = with_isolation(
            evaluate,
            vram_reader=vram_reader,
            pid_usage_reader=pid_usage_reader,
            gpu_sampler=gpu_sampler,
        )
    if storage is None and study_name:
        db_dir = base_config.data_dir / OPTUNA_METHOD
        db_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_dir / f'{study_name}.db'}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
        sampler=sampler,
    )
    study.optimize(
        make_objective(
            base_config,
            evaluate,
            model_spec=model_spec,
            vram_mib=vram_mib,
            ram_mib=ram_mib,
            on_trial=on_trial,
        ),
        n_trials=n_trials,
    )

    states = optuna.trial.TrialState
    complete = [t for t in study.trials if t.state == states.COMPLETE]
    pruned = [t for t in study.trials if t.state == states.PRUNED]
    if not complete:
        raise RuntimeError(f"tuning '{study_name}': no trial completed (all pruned/failed)")
    # Best = highest quality, tie-broken by higher measured throughput (the latency tie-break).
    best = max(complete, key=lambda t: (t.value or 0.0, t.user_attrs.get("throughput", 0.0)))
    best_value = float(best.value) if best.value is not None else 0.0
    best_overrides = best.user_attrs["overrides"]
    _LOG.info(
        "[tune] %s best quality=%.4f tput=%.1f over %d trials (%d pruned): %s",
        study_name,
        best_value,
        best.user_attrs.get("throughput", 0.0),
        len(study.trials),
        len(pruned),
        best_overrides,
    )
    return TuneResult(
        best_config=base_config.with_overrides(**best_overrides),
        best_value=best_value,
        n_trials=len(study.trials),
        n_complete=len(complete),
        n_pruned=len(pruned),
        study_name=study_name,
        storage=storage,
    )


def two_stage(
    base_config: RunConfig,
    *,
    n_trials: int,
    study_name: str,
    evaluate: Objective | None = None,
    final_runner: Callable[[RunConfig], EvalResult] | None = None,
    storage: str | None = None,
    seed: int = 13,
    model_spec: ModelSpec | None = None,
    vram_mib: int = 0,
    ram_mib: int = 0,
    on_trial: TrialCallback | None = None,
    isolate: bool = False,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
    gpu_sampler: Callable[[], list[Any]] | None = None,
) -> TwoStageResult:
    """Stage 1 tunes on the tuning split; stage 2 scores the winner on the full final split."""
    result = tune(
        base_config,
        n_trials=n_trials,
        study_name=study_name,
        evaluate=evaluate,
        storage=storage,
        seed=seed,
        model_spec=model_spec,
        vram_mib=vram_mib,
        ram_mib=ram_mib,
        on_trial=on_trial,
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
        gpu_sampler=gpu_sampler,
    )
    runner = final_runner or _run_eval_final
    _LOG.info(
        "[tune] %s stage-2 scoring the winning config on the '%s' split", study_name, FINAL_SPLIT
    )
    return TwoStageResult(tune=result, final=runner(result.best_config))


def _build_store(config: RunConfig) -> Any:
    from llb.rag.store import RagStore

    return RagStore.build(
        config.corpus_root,
        config.strategy,
        config.chunk_size,
        config.chunk_overlap,
        config.embedding_model,
        mode=config.retrieval_mode,
        child_size=config.child_chunk_size,
    )


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
