"""AvailabilityResolver: pick the backend that actually serves a model on THIS host (backend resolver).

The feasibility planner (`planner.plan.plan_model`) answers "does the model fit the VRAM+RAM
budget, and at what context". The resolver adds the two things on top of that:

  1. DISCOVERY -- does each backend's source actually exist / can it be served?
       - vllm:     the HF repo exists (Hugging Face Hub).
       - ollama:   the tag is pulled locally or resolvable in the Ollama library.
       - llamacpp: the (GGUF) repo exposes at least one `*.gguf` file.
     Plus the runtime itself: a source whose architecture the installed runtime is too old to
     implement is present and unservable, and becomes a NAMED skip (`runtime_floor`) rather than
     an anonymous "source not found".
  2. PRIORITY + FIT -- among the AVAILABLE sources, choose by the fixed backend priority
     vllm > ollama > llamacpp, but only when the chosen backend can actually run at the
     planner's verdict. This is fit-aware because the backends differ on CPU offload:
       - vLLM keeps the whole model in VRAM (no layer split to CPU), so it needs `gpu`.
       - Ollama / llama.cpp split layers GPU<->CPU RAM, so `offload` is fine for them.
     That encodes the design rule "prefer vLLM when the model fits VRAM, else fall back to
     the GGUF on Ollama/llama.cpp" (see `samples/configs/models_uk.yaml`).

A model declares its serving options either as the single `backend` + `source` it already
carries, or as a `sources: {backend: source}` map for the same logical model across backends.
Every probe is injectable, so the decision logic is pure and unit-testable without network.
"""

from llb.backends.planner.constants import VERDICT_GPU, VERDICT_NO, VERDICT_OFFLOAD
from llb.backends.planner.plan import plan_model
from llb.core.contracts.models import BackendCandidate, ModelPlanRow, ModelSpec, ResolvedModel
from llb.backends.resolver_probes import ResolverProbes, _probe_available
from llb.backends.runtime_floor import RUNTIME_FLOOR_SKIP, source_floor_reason
from llb.backends.resolver_sources import _priced_spec, candidate_sources
from llb.backends.resolver_feasibility import (
    MIN_SERVING_CTX,
    _plan_kwargs_for_backend,
    _plan_vram_for_backend,
    backend_fits,
)

# Fixed backend preference order (highest first). Sources for absent backends are skipped.

# A backend is "runnable" only if it can serve at least this many tokens of context. Judging
# fit at the host's MAX context would reject vLLM for any model that needs even one layer
# offloaded at the long end (e.g. gemma-4-E4B at 131072) yet serves fine at a normal context.


def resolve(
    spec: ModelSpec,
    vram_mib: int,
    ram_mib: int,
    *,
    probes: ResolverProbes | None = None,
    target_ctx: int | None = None,
    min_serving_ctx: int = MIN_SERVING_CTX,
    **plan_kwargs: object,
) -> ResolvedModel:
    """Resolve one logical model to a runnable (backend, source), or none with a reason."""
    probes = probes or ResolverProbes()
    candidates: list[BackendCandidate] = []
    chosen: BackendCandidate | None = None

    for backend, overrides in candidate_sources(spec):
        source = overrides["source"]
        # A source the runtime is too old to serve is present but unservable; it is a NAMED skip,
        # never "source not found" and never a generic backend error at launch.
        floor_skip = source_floor_reason(
            backend,
            source,
            {**spec, **overrides},
            version_reader=probes.runtime_version,
            requirement_reader=probes.artifact_requirement,
        )
        available = not floor_skip and _probe_available(backend, source, probes)
        backend_plan_kwargs = _plan_kwargs_for_backend(backend, dict(plan_kwargs))
        row = plan_model(
            _priced_spec(spec, backend, overrides),
            _plan_vram_for_backend(backend, vram_mib),
            ram_mib,
            target_ctx=target_ctx,
            **backend_plan_kwargs,  # type: ignore[arg-type]
        )
        verdict = row["verdict"]
        fits = backend_fits(backend, row, min_serving_ctx)
        runnable = available and fits
        candidate: BackendCandidate = {
            "backend": backend,
            "source": source,
            "quant": row["quant"],
            "gpu_layers": row["gpu_layers"],
            "available": available,
            "verdict": verdict,
            "runnable": runnable,
            "reason": floor_skip
            or _reason(available, backend, verdict, fits, row, min_serving_ctx),
        }
        if floor_skip:
            candidate["skip"] = RUNTIME_FLOOR_SKIP
        candidates.append(candidate)
        if runnable and chosen is None:  # first runnable wins (sources are priority-ordered)
            chosen = candidate

    return {
        "name": spec.get("name", spec.get("source", "?")),
        "chosen_backend": chosen["backend"] if chosen else None,
        "chosen_source": chosen["source"] if chosen else None,
        # vLLM is only ever chosen when a serving window fits fully on GPU, so report `gpu`
        # there rather than the planner's max-context verdict (which may say `offload`).
        "verdict": _serving_verdict(chosen) if chosen else VERDICT_NO,
        "candidates": candidates,
        "note": "" if chosen else "no available backend can serve this model on the host",
    }


def _serving_verdict(chosen: BackendCandidate) -> str:
    if chosen["backend"] == "vllm":
        return VERDICT_GPU
    return chosen["verdict"]


def llamacpp_offload_split(resolved: ResolvedModel) -> int | None:
    """The `-ngl` count to pass to llama.cpp for a resolved model, or None to keep the default.

    For an OFFLOAD verdict, return the planner's GPU/CPU layer split so the runner places only
    the layers that fit in VRAM on the GPU and lets the rest spill to system RAM -- instead of
    the launcher default (-1 == every layer on GPU), which would OOM an oversized model. Returns
    None when the chosen backend is not llama.cpp, when all layers fit on the GPU (the default -1
    is correct), or when the planner could not size a split (no arch fields)."""
    if resolved["chosen_backend"] != "llamacpp" or resolved["verdict"] != VERDICT_OFFLOAD:
        return None
    chosen = next((c for c in resolved["candidates"] if c["backend"] == "llamacpp"), None)
    if chosen is None:
        return None
    split = chosen.get("gpu_layers")
    return split if isinstance(split, int) and split > 0 else None


def _reason(
    available: bool, backend: str, verdict: str, fits: bool, row: ModelPlanRow, min_ctx: int
) -> str:
    if not available:
        return "source not found"
    if fits:
        ctx = row["ctx_gpu"] if backend == "vllm" else row["ctx_max"]
        return f"{verdict} -- runnable (ctx<={ctx})"
    if backend == "vllm" and row["ctx_max"] >= min_ctx:
        return f"only {row['ctx_gpu']} fits VRAM (< {min_ctx}); vLLM has no CPU offload -- use the GGUF"
    return f"{verdict} -- not runnable (< {min_ctx} ctx)"


def resolve_all(
    specs: list[ModelSpec],
    vram_mib: int,
    ram_mib: int,
    *,
    probes: ResolverProbes | None = None,
    target_ctx: int | None = None,
    min_serving_ctx: int = MIN_SERVING_CTX,
    **plan_kwargs: object,
) -> list[ResolvedModel]:
    probes = probes or ResolverProbes()
    return [
        resolve(
            s,
            vram_mib,
            ram_mib,
            probes=probes,
            target_ctx=target_ctx,
            min_serving_ctx=min_serving_ctx,
            **plan_kwargs,
        )
        for s in specs
    ]
