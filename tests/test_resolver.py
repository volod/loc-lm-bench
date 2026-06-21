"""AvailabilityResolver (M3.2): backend priority + offload-aware fit, driven by fakes."""

from llb.backends import resolver
from llb.backends.resolver import (
    ResolverProbes,
    backend_can_run,
    candidate_sources,
    format_resolution,
    resolve,
    resolve_all,
)
from llb.contracts import ModelSpec

ALL_AVAILABLE = ResolverProbes(
    hf_repo=lambda _s: True, gguf=lambda _s: True, ollama_tag=lambda _s: True
)

# A small model that fits a 16 GB card fully (planner verdict -> gpu).
SMALL: ModelSpec = {
    "name": "small",
    "backend": "vllm",
    "source": "org/small",
    "params_b": 3.0,
    "quant": "q4_k_m",
    "n_layers": 28,
    "kv_dim": 1024,
    "max_context": 8192,
}
# A big bf16 model that only runs by offloading layers to CPU RAM (verdict -> offload).
BIG: ModelSpec = {
    "name": "big",
    "backend": "vllm",
    "source": "org/big",
    "sources": {"vllm": "org/big", "ollama": "big:q4"},
    "params_b": 27.0,
    "quant": "bf16",
    "n_layers": 62,
    "kv_dim": 2048,
    "max_context": 8192,
}

HOST_VRAM = 16000
HOST_RAM = 64000


def test_backend_can_run_offload_semantics():
    assert backend_can_run("vllm", "gpu") is True
    assert backend_can_run("vllm", "offload") is False  # vLLM has no CPU offload
    assert backend_can_run("ollama", "offload") is True
    assert backend_can_run("llamacpp", "offload") is True
    assert backend_can_run("ollama", "no") is False


def test_candidate_sources_priority_ordered():
    spec: ModelSpec = {
        "name": "m",
        "backend": "ollama",
        "source": "m:tag",
        "sources": {"llamacpp": "org/m-gguf", "vllm": "org/m"},
    }
    # declared backend folded in; ordered vllm > ollama > llamacpp.
    assert candidate_sources(spec) == [
        ("vllm", "org/m"),
        ("ollama", "m:tag"),
        ("llamacpp", "org/m-gguf"),
    ]


def test_resolve_prefers_vllm_when_it_fits_vram():
    out = resolve(SMALL, HOST_VRAM, HOST_RAM, probes=ALL_AVAILABLE)
    assert out["chosen_backend"] == "vllm"
    assert out["verdict"] == "gpu"
    assert all(c["available"] for c in out["candidates"])


def test_resolve_falls_back_to_ollama_when_vllm_would_offload():
    out = resolve(BIG, HOST_VRAM, HOST_RAM, probes=ALL_AVAILABLE)
    # vLLM cannot offload to CPU, so the offload-verdict model resolves to Ollama.
    assert out["chosen_backend"] == "ollama"
    vllm = next(c for c in out["candidates"] if c["backend"] == "vllm")
    assert vllm["runnable"] is False and "offload" in vllm["reason"]


def test_resolve_marks_unavailable_source_not_runnable():
    probes = ResolverProbes(
        hf_repo=lambda _s: False,  # vLLM repo missing
        gguf=lambda _s: True,
        ollama_tag=lambda _s: True,
    )
    out = resolve(BIG, HOST_VRAM, HOST_RAM, probes=probes)
    vllm = next(c for c in out["candidates"] if c["backend"] == "vllm")
    assert vllm["available"] is False and vllm["runnable"] is False
    assert out["chosen_backend"] == "ollama"  # the available offload backend


def test_resolve_none_when_nothing_available():
    probes = ResolverProbes(
        hf_repo=lambda _s: False, gguf=lambda _s: False, ollama_tag=lambda _s: False
    )
    out = resolve(SMALL, HOST_VRAM, HOST_RAM, probes=probes)
    assert out["chosen_backend"] is None
    assert out["verdict"] == "no"
    assert "no available backend" in out["note"]


def test_resolve_all_and_format():
    rows = resolve_all([SMALL, BIG], HOST_VRAM, HOST_RAM, probes=ALL_AVAILABLE)
    table = format_resolution(rows)
    assert "small" in table and "big" in table
    assert "chosen" in table.splitlines()[0]


def test_ollama_probe_matches_bare_and_tagged(monkeypatch):
    body = '{"models": [{"name": "llama3.2:3b"}]}'

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return body.encode()

    monkeypatch.setattr(resolver.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    probe = resolver._make_ollama_probe("http://localhost:11434")
    assert probe("llama3.2:3b") is True
    assert probe("llama3.2") is True  # bare name matches :latest-style
    assert probe("mistral") is False
