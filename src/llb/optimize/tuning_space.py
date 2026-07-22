"""Focused tuning space implementation."""

from typing import Any, Callable, Sequence

from llb.backends.planner.plan import plan_model
from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec

TUNING_SPLIT = "tuning"

FINAL_SPLIT = "final"

STRATEGIES = ["fixed", "sentence", "recursive", "markdown", "semantic"]

EXTENDED_STRATEGIES = [*STRATEGIES, "page", "heading", "late"]

RETRIEVAL_MODES = ["flat", "parent_child", "hybrid"]

FUSION_WEIGHT_RANGE = (0.2, 0.8)

FUSION_CANDIDATES_RANGE = (20, 80)

GRAPH_WEIGHT_RANGE = (0.1, 0.5)

RERANK_CANDIDATES_RANGE = (15, 60)

# Token budgets that couple top_k, chunk_size, and max_model_len in multi-objective search.
CONTEXT_BUDGET_CHOICES = [2048, 4096, 8192, 16384]

CHARS_PER_TOKEN = 3.0  # UA measured ~0.33 tok/char in real-model validation -> ~3 chars/token

PROMPT_HEADROOM_TOKENS = 512  # system prompt + question + answer headroom

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


_OOM_MARKERS = ("out of memory", "outofmemory", "cuda error", "no available memory", "kv cache")

SERVING_MAX_MODEL_LEN = [4096, 8192, 16384]


def suggest_overrides(
    trial: Any,
    backend: str = "ollama",
    strategies: list[str] | None = None,
    reranker: str | None = None,
    embedders: Sequence[str] | None = None,
    tune_context_budget: bool = False,
    retrieval_backend: str = "faiss",
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
    sampling them for Ollama would tune dead parameters.
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
        overrides["gpu_memory_utilization"] = trial.suggest_float(
            "gpu_memory_utilization", 0.70, 0.90, step=0.05
        )
        # Context-budget couples max_model_len to the sampled token budget.
        if context_budget is not None:
            overrides["max_model_len"] = context_budget
        else:
            overrides["max_model_len"] = trial.suggest_categorical(
                "max_model_len", SERVING_MAX_MODEL_LEN
            )
    return overrides


def estimate_context_tokens(config: RunConfig, context_chars: int) -> int:
    """Rough tokens consumed by `context_chars` of context + headroom + the requested completion."""
    return int(context_chars / CHARS_PER_TOKEN) + PROMPT_HEADROOM_TOKENS + config.max_tokens


def estimate_prompt_tokens(config: RunConfig) -> int:
    """Rough tokens consumed by the retrieved context + headroom + the requested completion."""
    return estimate_context_tokens(config, config.top_k * config.chunk_size)


def effective_max_context(
    config: RunConfig, model_spec: ModelSpec, vram_mib: int, ram_mib: int
) -> int:
    """The smallest of: the planner's max context for the host, the model window, the
    served `max_model_len` cap, and an explicit `context_budget`. 0 means "cannot bound"."""
    row = plan_model(model_spec, vram_mib, ram_mib)
    ctx = row["ctx_max"] or int(model_spec.get("max_context") or 0)
    if config.max_model_len:
        ctx = min(ctx, config.max_model_len) if ctx else config.max_model_len
    if config.context_budget:
        ctx = min(ctx, config.context_budget) if ctx else config.context_budget
    return ctx


def fits_context_chars(
    config: RunConfig,
    model_spec: ModelSpec | None,
    vram_mib: int,
    ram_mib: int,
    context_chars: int,
) -> bool:
    """True if a prompt carrying `context_chars` of context fits the window / explicit budget.

    Without a `model_spec` only an explicit `context_budget` can bound the prompt, so an unknown
    model never silently declares a document unusable.
    """
    estimated = estimate_context_tokens(config, context_chars)
    if config.context_budget is not None and estimated > config.context_budget:
        return False
    if model_spec is None:
        return True
    ctx = effective_max_context(config, model_spec, vram_mib, ram_mib)
    return ctx <= 0 or estimated <= ctx


def fits_context(
    config: RunConfig, model_spec: ModelSpec | None, vram_mib: int, ram_mib: int
) -> bool:
    """True if the retrieved prompt fits the effective context / explicit budget."""
    return fits_context_chars(
        config, model_spec, vram_mib, ram_mib, config.top_k * config.chunk_size
    )


def is_oom(exc: BaseException) -> bool:
    """True for a MEASURED capacity failure, which every Optuna study prunes instead of crashing."""
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _OOM_MARKERS)
