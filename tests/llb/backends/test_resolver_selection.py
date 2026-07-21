"""Tests for resolver selection."""

from llb.backends import resolver
from llb.backends.resolver import (
    resolve,
    resolve_all,
)
from llb.backends.resolver_report import format_resolution
from llb.backends.resolver_sources import candidate_sources
from llb.core.contracts.models import ModelSpec
from test_resolver import ALL_AVAILABLE, BIG, HOST_RAM, HOST_VRAM, MISTRAL_MULTI, SMALL


def test_resolve_all_and_format():
    rows = resolve_all([SMALL, BIG], HOST_VRAM, HOST_RAM, probes=ALL_AVAILABLE)
    table = format_resolution(rows)
    assert "small" in table and "big" in table
    assert "chosen" in table.splitlines()[0]


def test_mistral_w4a16_resolves_vllm_on_24gb_and_gguf_below():
    # gpu-tier-24-mistral-vllm: the w4a16 vLLM source holds GPU-resident on a 24 GiB card, so the
    # resolver picks vLLM there; on a 16 GiB card it falls back to the curated Ollama GGUF.
    mistral: ModelSpec = {
        "name": "mistral-small-3.1-24b",
        "backend": "vllm",
        "source": "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
        "params_b": 24,
        "quant": "w4a16",
        "n_layers": 40,
        "kv_dim": 1024,
        "max_context": 131072,
        "vocab_size": 131072,
        "hidden_size": 5120,
        "tie_word_embeddings": False,
        "sources": {"ollama": {"source": "mistral-small3.1:24b", "quant": "q4_k_m"}},
    }
    at_24 = resolve(mistral, 24576, 128 * 1024, probes=ALL_AVAILABLE)
    assert at_24["chosen_backend"] == "vllm" and at_24["verdict"] == "gpu"
    at_16 = resolve(mistral, 16380, 128 * 1024, probes=ALL_AVAILABLE)
    assert at_16["chosen_backend"] == "ollama" and at_16["chosen_source"] == "mistral-small3.1:24b"


def test_candidate_sources_ranks_vllm_quants_by_quality():
    # The vLLM quants are ordered highest-bpw first, then the lower-priority backends follow.
    order = [(b, rec.get("quant")) for b, rec in candidate_sources(MISTRAL_MULTI)]
    assert order == [("vllm", "fp8"), ("vllm", "w4a16"), ("ollama", "q4_k_m")]


def test_candidate_sources_ranks_ollama_fallback_quants_by_quality():
    spec: ModelSpec = {
        **MISTRAL_MULTI,
        "sources": {
            "ollama": [
                {"source": "mistral:iq3", "quant": "iq3", "bpw": 3.5},
                {"source": "mistral:q4", "quant": "q4_k_m"},
            ]
        },
    }

    order = [
        (record["source"], record.get("quant"))
        for backend, record in candidate_sources(spec)
        if backend == "ollama"
    ]
    assert order == [("mistral:q4", "q4_k_m"), ("mistral:iq3", "iq3")]


def test_multi_quant_vllm_picks_best_fit_per_host():
    # resolver-multi-quant-vllm acceptance: highest-quality quant that fits GPU wins per tier --
    # GGUF on 16 GiB, w4a16 on 24 GiB, fp8 on 32 GiB -- so the sweep path now agrees with the
    # 32 GiB serving config (fp8) instead of resolving the smaller w4a16 everywhere.
    at_16 = resolve(MISTRAL_MULTI, 16380, 128 * 1024, probes=ALL_AVAILABLE)
    assert at_16["chosen_backend"] == "ollama" and at_16["chosen_source"] == "mistral-small3.1:24b"
    at_24 = resolve(MISTRAL_MULTI, 24576, 128 * 1024, probes=ALL_AVAILABLE)
    assert at_24["chosen_backend"] == "vllm" and at_24["verdict"] == "gpu"
    assert at_24["chosen_source"].endswith("quantized.w4a16")
    at_32 = resolve(MISTRAL_MULTI, 32607, 128 * 1024, probes=ALL_AVAILABLE)
    assert at_32["chosen_backend"] == "vllm" and at_32["verdict"] == "gpu"
    assert at_32["chosen_source"].endswith("FP8-dynamic")


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
