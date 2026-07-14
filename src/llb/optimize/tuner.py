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

from typing import Any, Callable

from llb.core.config import RunConfig
from llb.core.contracts import EvalResult, ModelSpec
from llb.optimize.tuning_space import (
    FINAL_SPLIT,
    Objective,
    estimate_prompt_tokens,
    fits_context,
    is_oom,
    suggest_overrides,
    with_isolation,
)
from llb.optimize.tuner_runtime import TrialCallback, _LOG, _run_eval_final, _run_eval_quality
from llb.optimize.tuner_models import TuneResult, TwoStageResult


OPTUNA_METHOD = "optuna"

# The corpus-chunking additions (page / heading / late) join the search space only behind an
# explicit flag (`tune --extended-chunkers`): `late` re-embeds whole documents per trial and
# `page` only differs from `recursive` on sidecar-bearing PDF corpora, so they are opt-in.
# Hybrid fusion search ranges (hybrid-retrieval-uk): the dense share of the weighted RRF and
# the per-side candidate depth, sampled only when the trial picked hybrid mode.
# Rerank search range (rerank-context-order): the candidate pool depth fed into the
# cross-encoder, sampled only when the trial turned the opt-in reranker on.

# config -> quality on the tuning split, OR (quality, throughput) for the latency tie-break.


# Substrings that mark a measured out-of-memory / capacity failure -> prune, do not crash.


def make_objective(
    base_config: RunConfig,
    evaluate: Objective,
    *,
    model_spec: ModelSpec | None = None,
    vram_mib: int = 0,
    ram_mib: int = 0,
    on_trial: TrialCallback | None = None,
    strategies: list[str] | None = None,
    reranker: str | None = None,
) -> Callable[[Any], float]:
    """Build the Optuna objective: sample -> validate -> prune over-context -> evaluate.

    The estimate-based context prune happens BEFORE a trial runs; a MEASURED OOM during the
    run (the model actually ran out of memory) prunes the trial too, so a too-aggressive
    serving config is dropped instead of crashing the whole study.
    """
    import optuna

    def objective(trial: Any) -> float:
        overrides = suggest_overrides(
            trial, backend=base_config.backend, strategies=strategies, reranker=reranker
        )
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
            if is_oom(exc):
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
    strategies: list[str] | None = None,
    reranker: str | None = None,
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
            strategies=strategies,
            reranker=reranker,
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
    strategies: list[str] | None = None,
    reranker: str | None = None,
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
        strategies=strategies,
        reranker=reranker,
    )
    runner = final_runner or _run_eval_final
    _LOG.info(
        "[tune] %s stage-2 scoring the winning config on the '%s' split", study_name, FINAL_SPLIT
    )
    return TwoStageResult(tune=result, final=runner(result.best_config))
