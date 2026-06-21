"""AvailabilityResolver: pick the backend that actually serves a model on THIS host (M3.2).

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
from typing import Callable

from llb.backends import planner
from llb.config import DEFAULT_OLLAMA_HOST
from llb.contracts import BackendCandidate, ModelPlanRow, ModelSpec, ResolvedModel

# Fixed backend preference order (highest first). Sources for absent backends are skipped.
BACKEND_PRIORITY = ("vllm", "ollama", "llamacpp")

# A backend is "runnable" only if it can serve at least this many tokens of context. Judging
# fit at the host's MAX context would reject vLLM for any model that needs even one layer
# offloaded at the long end (e.g. gemma-4-E4B at 131072) yet serves fine at a normal context.
MIN_SERVING_CTX = 2048

# Probes: source -> availability signal. Defaults hit HF Hub / Ollama; all injectable.
HfRepoProbe = Callable[[str], bool]  # repo id -> exists
GgufProbe = Callable[[str], bool]  # repo id -> has at least one *.gguf file
OllamaProbe = Callable[[str], bool]  # tag -> pulled locally or in the Ollama library


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


def candidate_sources(spec: ModelSpec) -> list[tuple[str, str]]:
    """The (backend, source) options for a spec, ordered by `BACKEND_PRIORITY`.

    Uses the explicit `sources` map when present, else the single declared backend+source.
    """
    declared: dict[str, str] = dict(spec.get("sources") or {})
    declared.setdefault(spec["backend"], spec["source"])
    return [(b, declared[b]) for b in BACKEND_PRIORITY if b in declared]


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

    for backend, source in candidate_sources(spec):
        available = _probe_available(backend, source, probes)
        row = planner.plan_model(
            {**spec, "backend": backend, "source": source},
            vram_mib,
            ram_mib,
            target_ctx=target_ctx,
            **plan_kwargs,  # type: ignore[arg-type]
        )
        verdict = row["verdict"]
        fits = backend_fits(backend, row, min_serving_ctx)
        runnable = available and fits
        candidate: BackendCandidate = {
            "backend": backend,
            "source": source,
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
