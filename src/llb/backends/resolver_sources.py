"""Focused resolver sources implementation."""

from typing import Any
from llb.backends.planner.weights import resolve_bpw
from llb.core.contracts.models import ModelSpec, SourceRecord

BACKEND_PRIORITY = ("vllm", "ollama", "llamacpp")


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
    bpw = resolve_bpw(_priced_spec(spec, "vllm", record))
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
