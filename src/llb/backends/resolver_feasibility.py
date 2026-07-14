"""Focused resolver feasibility implementation."""

from llb.backends.planner.constants import VERDICT_GPU, VERDICT_OFFLOAD
from llb.core.contracts import (
    ModelPlanRow,
)

MIN_SERVING_CTX = 2048

VLLM_RESOLUTION_GPU_MEMORY_UTILIZATION = 0.85


def _plan_kwargs_for_backend(backend: str, plan_kwargs: dict[str, object]) -> dict[str, object]:
    """Backend-specific planner knobs used only for availability resolution."""
    if backend != "vllm":
        return plan_kwargs
    from llb.executor.contention_memory import DEFAULT_VLLM_OVERHEAD_MB

    return {
        "vram_reserve": 0,
        "overhead_mib": DEFAULT_VLLM_OVERHEAD_MB,
        **plan_kwargs,
    }


def _plan_vram_for_backend(backend: str, vram_mib: int) -> int:
    """Effective backend allocation budget for availability resolution."""
    if backend != "vllm":
        return vram_mib
    return int(vram_mib * VLLM_RESOLUTION_GPU_MEMORY_UTILIZATION)


def backend_can_run(backend: str, verdict: str) -> bool:
    """Can `backend` serve a model the planner gave this `verdict`? (weight-only fallback)

    vLLM has no CPU offload, so it needs the model fully in VRAM (`gpu`). Ollama and
    llama.cpp split layers to CPU RAM, so `offload` still runs (slower). Used only when the
    spec lacks the architecture fields to size a KV cache; otherwise `backend_fits` is sharper.
    """
    if backend == "vllm":
        return verdict == VERDICT_GPU
    if backend in ("ollama", "llamacpp"):
        return verdict in (VERDICT_GPU, VERDICT_OFFLOAD)
    return False


def backend_fits(backend: str, row: ModelPlanRow, min_ctx: int = MIN_SERVING_CTX) -> bool:
    """Can `backend` serve this planned model at >= `min_ctx` tokens of context?

    vLLM must hold a `min_ctx` window fully on GPU (`ctx_gpu`); Ollama / llama.cpp may use
    GPU+CPU offload (`ctx_max`). Falls back to the verdict when the spec has no architecture
    to size the KV cache (`ctx_max == 0`).
    """
    if row["ctx_max"] <= 0:
        return backend_can_run(backend, row["verdict"])
    if backend == "vllm":
        return row["ctx_gpu"] >= min_ctx
    if backend in ("ollama", "llamacpp"):
        return row["ctx_max"] >= min_ctx
    return False
