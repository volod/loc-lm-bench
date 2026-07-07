"""AvailabilityResolver: pick the backend that actually serves a model on THIS host (backend resolver).

The feasibility planner (`planner.plan_model`) answers "does the model fit the VRAM+RAM
budget, and at what context". The resolver adds the two things on top of that:

  1. DISCOVERY -- does each backend's source actually exist / can it be served?
       - vllm:     the HF repo exists (Hugging Face Hub).
       - ollama:   the tag is pulled locally or resolvable in the Ollama library.
       - llamacpp: the (GGUF) repo exposes at least one `*.gguf` file.
  2. PRIORITY + FIT -- among the AVAILABLE sources, choose by the fixed backend priority
     vllm > ollama > llamacpp, but only when the chosen backend can actually run at the
     planner's verdict. This is fit-aware because the backends differ on CPU offload:
       - vLLM keeps the whole model in VRAM (no layer split to CPU), so it needs `gpu`.
       - Ollama / llama.cpp split layers GPU<->CPU RAM, so `offload` is fine for them.
     That encodes the design rule "prefer vLLM when the model fits VRAM, else fall back to
     the GGUF on Ollama/llama.cpp" (see `samples/models_uk.yaml`).

A model declares its serving options either as the single `backend` + `source` it already
carries, or as a `sources: {backend: source}` map for the same logical model across backends.
Every probe is injectable, so the decision logic is pure and unit-testable without network.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from llb.backends import planner
from llb.core.config import DEFAULT_OLLAMA_HOST
from llb.core.contracts import BackendCandidate, ModelPlanRow, ModelSpec, ResolvedModel, SourceRecord

# Fixed backend preference order (highest first). Sources for absent backends are skipped.
BACKEND_PRIORITY = ("vllm", "ollama", "llamacpp")

# A backend is "runnable" only if it can serve at least this many tokens of context. Judging
# fit at the host's MAX context would reject vLLM for any model that needs even one layer
# offloaded at the long end (e.g. gemma-4-E4B at 131072) yet serves fine at a normal context.
MIN_SERVING_CTX = 2048
VLLM_RESOLUTION_GPU_MEMORY_UTILIZATION = 0.85

# Probes: source -> availability signal. Defaults hit HF Hub / Ollama; all injectable.
HfRepoProbe = Callable[[str], bool]  # repo id -> exists
GgufProbe = Callable[[str], bool]  # repo id -> has at least one *.gguf file
OllamaProbe = Callable[[str], bool]  # tag -> pulled locally or in the Ollama library


def _plan_kwargs_for_backend(backend: str, plan_kwargs: dict[str, object]) -> dict[str, object]:
    """Backend-specific planner knobs used only for availability resolution."""
    if backend != "vllm":
        return plan_kwargs
    from llb.executor.contention import DEFAULT_VLLM_OVERHEAD_MB

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
        return verdict == planner.VERDICT_GPU
    if backend in ("ollama", "llamacpp"):
        return verdict in (planner.VERDICT_GPU, planner.VERDICT_OFFLOAD)
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


def normalize_source(value: "str | SourceRecord") -> dict[str, Any]:
    """A source entry is either a bare source string or a record with metadata overrides."""
    if isinstance(value, str):
        return {"source": value}
    return {k: v for k, v in dict(value).items() if v is not None}


def normalize_source_list(value: Any) -> list[dict[str, Any]]:
    """A backend's `sources` value is one source (str/record) or a LIST of them (multiple quants)."""
    if isinstance(value, list):
        return [normalize_source(v) for v in value]
    return [normalize_source(value)]


def _quant_quality(spec: ModelSpec, record: dict[str, Any]) -> float:
    """Rank key for competing same-backend quants: higher bits-per-weight = higher quality."""
    bpw = planner.resolve_bpw(_priced_spec(spec, "vllm", record))
    return bpw if bpw is not None else -1.0


def candidate_sources(spec: ModelSpec) -> list[tuple[str, dict[str, Any]]]:
    """The (backend, source-record) options for a spec, ordered by `BACKEND_PRIORITY`.

    Each record carries at least `source` plus any per-artifact overrides (quant, arch, gating)
    so the planner prices the real artifact. The declared backend folds in the spec-level source
    (its quant/arch already live on the spec). A backend may declare a LIST of sources -- several
    vLLM quants of one model -- in which case they are ordered highest-quality first, so the
    "first runnable wins" rule below picks the best quant that fits the host on GPU (fp8 on a 32 GiB
    card, w4a16 on a 24 GiB card) before falling through to the Ollama/llama.cpp offload.
    """
    declared: dict[str, list[dict[str, Any]]] = {
        b: normalize_source_list(v) for b, v in (spec.get("sources") or {}).items()
    }
    declared.setdefault(spec["backend"], [{"source": spec["source"]}])
    out: list[tuple[str, dict[str, Any]]] = []
    for backend in BACKEND_PRIORITY:
        records = declared.get(backend)
        if not records:
            continue
        if backend == "vllm" and len(records) > 1:
            records = sorted(records, key=lambda r: _quant_quality(spec, r), reverse=True)
        out.extend((backend, record) for record in records)
    return out


def _priced_spec(spec: ModelSpec, backend: str, overrides: dict[str, Any]) -> ModelSpec:
    """The spec the planner should price for one candidate: parent fields + per-source overrides."""
    return {**spec, "backend": backend, **overrides}  # type: ignore[typeddict-item]


def _probe_available(backend: str, source: str, probes: "ResolverProbes") -> bool:
    if backend == "vllm":
        return probes.hf_repo(source)
    if backend == "ollama":
        return probes.ollama_tag(source)
    if backend == "llamacpp":
        return probes.gguf(source)
    return False


class ResolverProbes:
    """The three availability probes, defaulting to live HF Hub / Ollama checks."""

    def __init__(
        self,
        hf_repo: HfRepoProbe | None = None,
        gguf: GgufProbe | None = None,
        ollama_tag: OllamaProbe | None = None,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
    ):
        self.hf_repo = hf_repo or _hf_repo_exists
        self.gguf = gguf or _hf_has_gguf
        self.ollama_tag = ollama_tag or _make_ollama_probe(ollama_host)


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
        available = _probe_available(backend, source, probes)
        backend_plan_kwargs = _plan_kwargs_for_backend(backend, dict(plan_kwargs))
        row = planner.plan_model(
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
            "reason": _reason(available, backend, verdict, fits, row, min_serving_ctx),
        }
        candidates.append(candidate)
        if runnable and chosen is None:  # first runnable wins (sources are priority-ordered)
            chosen = candidate

    return {
        "name": spec.get("name", spec.get("source", "?")),
        "chosen_backend": chosen["backend"] if chosen else None,
        "chosen_source": chosen["source"] if chosen else None,
        # vLLM is only ever chosen when a serving window fits fully on GPU, so report `gpu`
        # there rather than the planner's max-context verdict (which may say `offload`).
        "verdict": _serving_verdict(chosen) if chosen else planner.VERDICT_NO,
        "candidates": candidates,
        "note": "" if chosen else "no available backend can serve this model on the host",
    }


def _serving_verdict(chosen: BackendCandidate) -> str:
    if chosen["backend"] == "vllm":
        return planner.VERDICT_GPU
    return chosen["verdict"]


def llamacpp_offload_split(resolved: ResolvedModel) -> int | None:
    """The `-ngl` count to pass to llama.cpp for a resolved model, or None to keep the default.

    For an OFFLOAD verdict, return the planner's GPU/CPU layer split so the runner places only
    the layers that fit in VRAM on the GPU and lets the rest spill to system RAM -- instead of
    the launcher default (-1 == every layer on GPU), which would OOM an oversized model. Returns
    None when the chosen backend is not llama.cpp, when all layers fit on the GPU (the default -1
    is correct), or when the planner could not size a split (no arch fields)."""
    if resolved["chosen_backend"] != "llamacpp" or resolved["verdict"] != planner.VERDICT_OFFLOAD:
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


# --- live probes (best-effort; any error -> "not available", never raises) ----------------


def _hf_repo_exists(repo_id: str) -> bool:
    try:
        from huggingface_hub import HfApi

        return bool(HfApi().repo_exists(repo_id))
    except Exception:
        return False


def _hf_has_gguf(repo_id: str) -> bool:
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo_id)
        return any(f.lower().endswith(".gguf") for f in files)
    except Exception:
        return False


def _make_ollama_probe(host: str) -> OllamaProbe:
    def probe(tag: str) -> bool:
        try:
            url = f"{host.rstrip('/')}/api/tags"
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                body = json.loads(resp.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError):
            return False
        names = {m.get("name", "") for m in body.get("models", [])}
        # Match `llama3.2:3b` and a bare `llama3.2` (Ollama defaults to :latest).
        return tag in names or any(n.split(":", 1)[0] == tag.split(":", 1)[0] for n in names)

    return probe


def format_resolution(rows: list[ResolvedModel]) -> str:
    """ASCII table: the chosen backend per model + the verdict."""
    headers = ["model", "chosen", "source", "verdict", "note"]

    def fmt(r: ResolvedModel) -> list[str]:
        return [
            r["name"],
            r["chosen_backend"] or "-",
            r["chosen_source"] or "-",
            r["verdict"],
            r["note"] or "ok",
        ]

    table = [fmt(r) for r in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in table)) if table else len(h) for i, h in enumerate(headers)
    ]
    out = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for r in table:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(out)
