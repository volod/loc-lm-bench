"""Two-stage Optuna tuning of backend + RAG parameters (M3.4).

The split discipline keeps the leaderboard honest: STAGE 1 searches the RAG/backend space on
the disjoint `tuning` split (a proxy -- never the final gold items), STAGE 2 scores ONLY the
winning config on the full `final` split, and only that stage-2 run is the leaderboard entry.
The embedding is PINNED (Premise 4) and is never a search dimension.

Search space (the chunking machinery already exists in M1):
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
from llb.config import RunConfig
from llb.contracts import EvalResult, ModelSpec

_LOG = logging.getLogger(__name__)

TUNING_SPLIT = "tuning"
FINAL_SPLIT = "final"
OPTUNA_METHOD = "optuna"

STRATEGIES = ["fixed", "sentence", "recursive", "markdown", "semantic"]
RETRIEVAL_MODES = ["flat", "parent_child"]
CHARS_PER_TOKEN = 3.0  # UA measured ~0.33 tok/char in M2.4 -> ~3 chars/token
PROMPT_HEADROOM_TOKENS = 512  # system prompt + question + answer headroom

Objective = Callable[[RunConfig], float]  # config -> quality on the tuning split


def suggest_overrides(trial: Any) -> dict[str, Any]:
    """Sample one RAG config from an Optuna trial (embedding is pinned, never sampled)."""
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


def make_objective(
    base_config: RunConfig,
    evaluate: Objective,
    *,
    model_spec: ModelSpec | None = None,
    vram_mib: int = 0,
    ram_mib: int = 0,
) -> Callable[[Any], float]:
    """Build the Optuna objective: sample -> validate -> prune over-context -> evaluate."""
    import optuna

    def objective(trial: Any) -> float:
        overrides = suggest_overrides(trial)
        try:
            config = base_config.with_overrides(**overrides)
        except ValueError as exc:  # e.g. overlap >= chunk_size after rounding
            raise optuna.TrialPruned(f"invalid config: {exc}") from None
        if not fits_context(config, model_spec, vram_mib, ram_mib):
            raise optuna.TrialPruned(
                f"retrieved context ~{estimate_prompt_tokens(config)} tok exceeds the model window"
            )
        trial.set_user_attr("overrides", overrides)
        return evaluate(config)

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
) -> TuneResult:
    """Stage 1: search the RAG/backend space on the tuning split; return the best config.

    `storage` defaults to a persistent SQLite study under ``$DATA_DIR/optuna/`` so a killed
    run resumes (set `storage=None` only in tests for an in-memory study).
    """
    import optuna

    evaluate = evaluate or _run_eval_quality
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
            base_config, evaluate, model_spec=model_spec, vram_mib=vram_mib, ram_mib=ram_mib
        ),
        n_trials=n_trials,
    )

    states = optuna.trial.TrialState
    complete = [t for t in study.trials if t.state == states.COMPLETE]
    pruned = [t for t in study.trials if t.state == states.PRUNED]
    if not complete:
        raise RuntimeError(f"tuning '{study_name}': no trial completed (all pruned/failed)")
    best_overrides = study.best_trial.user_attrs["overrides"]
    _LOG.info(
        "[tune] %s best quality=%.4f over %d trials (%d pruned): %s",
        study_name,
        study.best_value,
        len(study.trials),
        len(pruned),
        best_overrides,
    )
    return TuneResult(
        best_config=base_config.with_overrides(**best_overrides),
        best_value=float(study.best_value),
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


def _run_eval_quality(config: RunConfig) -> float:
    """Default stage-1 objective: build the config's store, score the tuning split, take quality."""
    from llb.executor.runner import run_eval

    result = run_eval(config, store=_build_store(config), split=TUNING_SPLIT, emit=False)
    rows = result["rows"]
    return float(rows[0]["quality"]) if rows else 0.0


def _run_eval_final(config: RunConfig) -> EvalResult:
    """Default stage-2 run: score the winning config on the full final split (the entry)."""
    from llb.executor.runner import run_eval

    return run_eval(config, store=_build_store(config), split=FINAL_SPLIT, emit=True)
