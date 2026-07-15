"""AvailabilityResolver (backend resolver): backend priority + offload-aware fit, driven by fakes."""

from llb.backends.resolver import (
    ResolverProbes,
    llamacpp_offload_split,
    resolve,
)
from llb.backends.resolver_feasibility import backend_can_run
from llb.backends.resolver_sources import candidate_sources
from llb.core.contracts.models import ModelSpec

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
    # declared backend folded in; ordered vllm > ollama > llamacpp; each a source record.
    assert candidate_sources(spec) == [
        ("vllm", {"source": "org/m"}),
        ("ollama", {"source": "m:tag"}),
        ("llamacpp", {"source": "org/m-gguf"}),
    ]


def test_per_source_quant_is_priced_independently():
    # bf16 vLLM weights do not fit 16 GB, but the per-source q4 GGUF runs on Ollama (offload).
    spec: ModelSpec = {
        "name": "ua12b",
        "backend": "vllm",
        "source": "org/ua-bf16",
        "params_b": 12.0,
        "quant": "bf16",
        "n_layers": 48,
        "kv_dim": 2048,
        "max_context": 8192,
        "sources": {
            "ollama": {"source": "hf.co/ua-gguf:Q4_K_M", "quant": "q4_k_m", "min_vram_gb": 8},
        },
    }
    out = resolve(spec, HOST_VRAM, HOST_RAM, probes=ALL_AVAILABLE)
    by_backend = {c["backend"]: c for c in out["candidates"]}
    assert by_backend["vllm"]["quant"] == "bf16"  # priced as the bf16 artifact
    assert by_backend["ollama"]["quant"] == "q4_k_m"  # priced as the q4 GGUF, not vLLM metadata
    assert out["chosen_backend"] == "ollama"


def test_resolve_prefers_vllm_when_it_fits_vram():
    out = resolve(SMALL, HOST_VRAM, HOST_RAM, probes=ALL_AVAILABLE)
    assert out["chosen_backend"] == "vllm"
    assert out["verdict"] == "gpu"
    assert all(c["available"] for c in out["candidates"])


def test_vllm_resolution_accounts_for_serving_overhead_on_12gb():
    gemma_e4b: ModelSpec = {
        "name": "gemma-4-e4b-it-w4a16",
        "backend": "vllm",
        "source": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "params_b": 8,
        "quant": "w4a16",
        "n_layers": 42,
        "kv_dim": 512,
        "max_context": 131072,
        "hi_precision_params_b": 4.2,
    }

    out = resolve(gemma_e4b, 12227, HOST_RAM, probes=ALL_AVAILABLE)

    vllm = next(c for c in out["candidates"] if c["backend"] == "vllm")
    assert out["chosen_backend"] is None
    assert vllm["runnable"] is False
    assert "vLLM has no CPU offload" in vllm["reason"]


def test_vllm_resolution_accounts_for_default_memory_fraction_on_12gb():
    gemma_12b: ModelSpec = {
        "name": "gemma-4-12b-it-w4a16",
        "backend": "vllm",
        "source": "google/gemma-4-12B-it-qat-w4a16-ct",
        "params_b": 12,
        "quant": "w4a16",
        "n_layers": 48,
        "kv_dim": 2048,
        "max_context": 262144,
        "vocab_size": 262144,
        "hidden_size": 3840,
        "tie_word_embeddings": True,
    }

    out = resolve(gemma_12b, 12227, HOST_RAM, probes=ALL_AVAILABLE)

    vllm = next(c for c in out["candidates"] if c["backend"] == "vllm")
    assert out["chosen_backend"] is None
    assert vllm["runnable"] is False


def test_resolve_falls_back_to_ollama_when_vllm_would_offload():
    out = resolve(BIG, HOST_VRAM, HOST_RAM, probes=ALL_AVAILABLE)
    # vLLM cannot offload to CPU, so the offload-verdict model resolves to Ollama.
    assert out["chosen_backend"] == "ollama"
    vllm = next(c for c in out["candidates"] if c["backend"] == "vllm")
    assert vllm["runnable"] is False and "offload" in vllm["reason"]


# A GGUF-only model too big to hold fully in 16 GB VRAM at q4 -> resolves to llama.cpp by
# offloading some layers to CPU RAM (the llama.cpp launcher run-path: -ngl derived from the planner's split).
BIG_GGUF: ModelSpec = {
    "name": "big-gguf",
    "backend": "llamacpp",
    "source": "hf.co/org/Big-GGUF:Q4_K_M",
    "params_b": 27.0,
    "quant": "q4_k_m",
    "n_layers": 62,
    "kv_dim": 2048,
    "max_context": 8192,
}
GGUF_ONLY = ResolverProbes(
    hf_repo=lambda _s: False, gguf=lambda _s: True, ollama_tag=lambda _s: False
)


def test_candidate_carries_planner_gpu_layers_split():
    out = resolve(BIG_GGUF, HOST_VRAM, HOST_RAM, probes=GGUF_ONLY)
    cand = next(c for c in out["candidates"] if c["backend"] == "llamacpp")
    # The planner offloads only some layers (an oversized model), so 0 < split < n_layers.
    assert 0 < cand["gpu_layers"] < BIG_GGUF["n_layers"]


def test_offload_split_returns_planner_layers_for_llamacpp_offload():
    out = resolve(BIG_GGUF, HOST_VRAM, HOST_RAM, probes=GGUF_ONLY)
    assert out["chosen_backend"] == "llamacpp" and out["verdict"] == "offload"
    cand = next(c for c in out["candidates"] if c["backend"] == "llamacpp")
    # The runner should pin -ngl to the planner's split, not the launcher default (-1 = all GPU).
    assert llamacpp_offload_split(out) == cand["gpu_layers"]


def test_offload_split_is_none_when_model_fits_gpu():
    # A small GGUF fits fully (gpu verdict) -> keep the launcher default (-1 = all layers on GPU).
    small_gguf: ModelSpec = {
        **SMALL,
        "backend": "llamacpp",
        "source": "hf.co/org/Small-GGUF:Q4_K_M",
    }
    out = resolve(small_gguf, HOST_VRAM, HOST_RAM, probes=GGUF_ONLY)
    assert out["chosen_backend"] == "llamacpp" and out["verdict"] == "gpu"
    assert llamacpp_offload_split(out) is None


def test_offload_split_is_none_for_non_llamacpp_backend():
    out = resolve(BIG, HOST_VRAM, HOST_RAM, probes=ALL_AVAILABLE)  # resolves to Ollama (offload)
    assert out["chosen_backend"] == "ollama"
    assert llamacpp_offload_split(out) is None


# --- resolver-multi-quant-vllm: several vLLM quants of one model, best GPU fit wins ---------------

MISTRAL_MULTI: ModelSpec = {
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
    "sources": {
        "vllm": [
            {"source": "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic", "quant": "fp8"},
            {
                "source": "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
                "quant": "w4a16",
            },
        ],
        "ollama": {"source": "mistral-small3.1:24b", "quant": "q4_k_m"},
    },
}
