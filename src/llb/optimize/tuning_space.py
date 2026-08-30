"""The tuning search space: what a trial may sample, and what disqualifies it.

Pure and import-light (no `optuna`), so the ranges, the context-fit prune, and the OOM
classification are unit-testable on their own. `llb.optimize.tuner` drives them.

The window arithmetic a prune rests on is NOT here -- it is `llb.backends.context_fit`, beside the
launchers it asks about, so a sweep, an eval run, and an agent loop on one host cannot disagree
about what fits. What is here is the RETRIEVED shape of it: `top_k x chunk_size` is a search-space
quantity, and pricing it is the prune.
"""

import math
from typing import Any, Callable, Sequence

from llb.backends.context_fit import estimate_context_tokens, fits_context_chars
from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec

TUNING_SPLIT = "tuning"

FINAL_SPLIT = "final"

STRATEGIES = ["fixed", "sentence", "recursive", "markdown", "semantic"]

# The corpus-chunking additions join the search space only behind an explicit flag
# (`tune --extended-chunkers`): `late` re-embeds whole documents per trial, `page` only
# differs from `recursive` on sidecar-bearing PDF corpora, and `table` only differs on
# corpora carrying markdown tables, so they are opt-in.
EXTENDED_STRATEGIES = [*STRATEGIES, "page", "heading", "late", "table"]

RETRIEVAL_MODES = ["flat", "parent_child", "hybrid"]

# Hybrid fusion search ranges (hybrid-retrieval-uk): the dense share of the weighted RRF and
# the per-side candidate depth, sampled only when the trial picked hybrid mode.
FUSION_WEIGHT_RANGE = (0.2, 0.8)

FUSION_CANDIDATES_RANGE = (20, 80)

GRAPH_WEIGHT_RANGE = (0.1, 0.5)

# Rerank search range (rerank-context-order): the candidate pool depth fed into the
# cross-encoder, sampled only when the trial turned the opt-in reranker on.
RERANK_CANDIDATES_RANGE = (15, 60)

# Token budgets that couple top_k, chunk_size, and max_model_len in multi-objective search.
CONTEXT_BUDGET_CHOICES = [2048, 4096, 8192, 16384]

# float | (quality, throughput) | TrialMetrics-shaped outcomes from evaluate hooks.
Objective = Callable[[RunConfig], Any]


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

    def run(config: RunConfig) -> Any:
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

# vLLM serving-utilization search range. The upper bound is a HOST question, not a search
# question: a trial served above the utilization the run declared is a launch the operator never
# authorized, and on a card whose validated value is 0.80 it is an out-of-memory graph capture
# rather than a slow trial. So the declared `RunConfig.gpu_memory_utilization` is carried in as a
# CEILING (`gpu_memory_utilization_ceiling`) and the range is clamped under it.
GPU_MEMORY_UTILIZATION_RANGE = (0.70, 0.90)

GPU_MEMORY_UTILIZATION_STEP = 0.05


def suggest_gpu_memory_utilization(trial: Any, ceiling: float | None = None) -> float:
    """Sample the vLLM memory fraction, never above the run's declared serving ceiling.

    A ceiling at or below the range floor pins the value instead of sampling: there is no
    authorized band left to search, and pinning keeps the trial runnable rather than pruning it.
    """
    low, high = GPU_MEMORY_UTILIZATION_RANGE
    if ceiling is not None:
        high = min(high, ceiling)
    if high <= low:
        return high
    # Snap the top of the band down onto the step grid so the sampled value is always a grid
    # point AND never above the ceiling. The rounding is binary-float hygiene: 0.05 has no exact
    # representation, so the raw quotient of an exact grid width lands just under the integer.
    steps = math.floor(round((high - low) / GPU_MEMORY_UTILIZATION_STEP, 6))
    if steps < 1:
        return low
    sampled = trial.suggest_float(
        "gpu_memory_utilization",
        low,
        round(low + steps * GPU_MEMORY_UTILIZATION_STEP, 6),
        step=GPU_MEMORY_UTILIZATION_STEP,
    )
    # Optuna returns the grid point with its own float noise (0.7999999999999999 for 0.80). That
    # value reaches a `vllm serve` argv and a run manifest, so round it back onto the grid.
    return round(float(sampled), 4)


def suggest_overrides(
    trial: Any,
    backend: str = "ollama",
    strategies: list[str] | None = None,
    reranker: str | None = None,
    embedders: Sequence[str] | None = None,
    tune_context_budget: bool = False,
    retrieval_backend: str = "faiss",
    gpu_memory_utilization_ceiling: float | None = None,
) -> dict[str, Any]:
    """Sample one config from an Optuna trial.

    RAG params are always sampled; `strategies` overrides the chunking-strategy choices
    (`EXTENDED_STRATEGIES` behind `tune --extended-chunkers`). `reranker` (a cross-encoder id,
    `tune --reranker`) adds the opt-in rerank-context-order axes: reranker on/off plus the
    candidate depth, sampled only when on (dead parameters otherwise). `embedders` promotes the
    embedding model from a pinned constant to a categorical knob (multi-objective-rag-tuner).
    `tune_context_budget` samples a token budget that couples `top_k` / `chunk_size` /
    `max_model_len`. BACKEND-AWARE serving knobs are sampled only when the resolved backend
    actually exposes them: `gpu_memory_utilization` / `max_model_len` are vLLM concepts, so
    sampling them for Ollama would tune dead parameters, and `gpu_memory_utilization` is
    additionally clamped under `gpu_memory_utilization_ceiling` (the run's declared serving
    utilization) so no trial launches above what the host was validated for.
    """
    strategy = trial.suggest_categorical("strategy", list(strategies or STRATEGIES))
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
    if embedders:
        overrides["embedding_model"] = trial.suggest_categorical("embedding_model", list(embedders))
    if mode == "parent_child":
        # child must stay below chunk_size (and the validator wants overlap < child_size).
        ceiling = max(128, chunk_size - 64)
        child = trial.suggest_int("child_chunk_size", 128, 640, step=32)
        overrides["child_chunk_size"] = min(child, ceiling)
    if mode == "hybrid":
        # Fusion knobs only exist in hybrid mode (dead parameters otherwise).
        overrides["fusion_weight"] = trial.suggest_float(
            "fusion_weight", *FUSION_WEIGHT_RANGE, step=0.1
        )
        overrides["fusion_candidates"] = trial.suggest_int(
            "fusion_candidates", *FUSION_CANDIDATES_RANGE, step=20
        )
    if retrieval_backend == "fused":
        overrides["graph_weight"] = trial.suggest_float(
            "graph_weight", *GRAPH_WEIGHT_RANGE, step=0.1
        )
    if reranker is not None and trial.suggest_categorical("use_reranker", [False, True]):
        overrides["reranker"] = reranker
        overrides["rerank_candidates"] = trial.suggest_int(
            "rerank_candidates", *RERANK_CANDIDATES_RANGE, step=15
        )
    context_budget: int | None = None
    if tune_context_budget:
        context_budget = int(
            trial.suggest_categorical("context_budget", list(CONTEXT_BUDGET_CHOICES))
        )
        overrides["context_budget"] = context_budget
    if backend == "vllm":
        overrides["gpu_memory_utilization"] = suggest_gpu_memory_utilization(
            trial, gpu_memory_utilization_ceiling
        )
        # Context-budget couples max_model_len to the sampled token budget.
        if context_budget is not None:
            overrides["max_model_len"] = context_budget
        else:
            overrides["max_model_len"] = trial.suggest_categorical(
                "max_model_len", SERVING_MAX_MODEL_LEN
            )
    return overrides


def estimate_prompt_tokens(config: RunConfig) -> int:
    """Rough tokens consumed by the retrieved context + headroom + the requested completion."""
    return estimate_context_tokens(config, config.top_k * config.chunk_size)


def fits_context(
    config: RunConfig,
    model_spec: ModelSpec | None,
    vram_mib: int,
    ram_mib: int,
    served_max_model_len: int | None = None,
) -> bool:
    """True if the retrieved prompt fits the usable window / explicit budget."""
    return fits_context_chars(
        config,
        model_spec,
        vram_mib,
        ram_mib,
        config.top_k * config.chunk_size,
        served_max_model_len,
    )


def is_oom(exc: BaseException) -> bool:
    """True for a MEASURED capacity failure, which every Optuna study prunes instead of crashing."""
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _OOM_MARKERS)
