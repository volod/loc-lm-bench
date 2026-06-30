import json

import llb.backends.planner as planner
from llb.backends.hardware import parse_meminfo
from llb.backends.planner import (
    VERDICT_GPU,
    VERDICT_NO,
    VERDICT_OFFLOAD,
    VERDICT_UNKNOWN,
    arch_from_config,
    attention_layer_split,
    embedding_params,
    enrich_arch,
    hi_precision_params,
    kv_mib_at_context,
    kv_mib_per_token,
    max_context,
    max_context_for_kv,
    plan_model,
    resolve_bpw,
    weights_mib,
    weights_mib_detailed,
)

# A small model that fits comfortably on a 16 GB card.
SMALL = {
    "name": "small",
    "backend": "ollama",
    "params_b": 3.2,
    "quant": "q4_k_m",
    "n_layers": 28,
    "kv_dim": 1024,
    "max_context": 4096,
}
# A 7B in fp16: weights ~14.5 GB, so it overflows VRAM and needs CPU offload.
FP16_7B = {
    "name": "big",
    "backend": "vllm",
    "params_b": 7.6,
    "quant": "fp16",
    "n_layers": 28,
    "kv_dim": 512,
    "max_context": 8192,
}


def test_parse_meminfo():
    text = "MemTotal:       32791234 kB\nMemFree:  100 kB\n"
    assert parse_meminfo(text) == 32791234 // 1024


def test_resolve_bpw_table_and_explicit():
    assert resolve_bpw({"quant": "fp16"}) == 16.0
    assert resolve_bpw({"quant": "Q4_K_M"}) == 4.5
    assert resolve_bpw({"quant": "fp8"}) == 8.0  # served formats
    assert resolve_bpw({"quant": "w4a16"}) == 4.5  # int4 for vLLM
    assert resolve_bpw({"quant": "iq3"}) == 3.5
    assert resolve_bpw({"bpw": 5.0}) == 5.0
    assert resolve_bpw({"quant": "nonsense"}) is None


def test_weights_and_kv_math():
    assert round(weights_mib(3.2, 4.5)) == 1717  # 3.2e9 * 4.5 / 8 bytes -> MiB
    assert round(kv_mib_per_token(28, 1024), 4) == 0.1094  # 2*28*1024*2 bytes/token


def test_max_context_caps_and_floors():
    # plenty of budget -> capped at the model max
    assert max_context(100000, 1717, 512, 0.1094, cap=4096) == 4096
    # weights alone exceed the budget -> 0
    assert max_context(1000, 1717, 512, 0.1094, cap=4096) == 0


def test_small_model_fits_fully_on_gpu():
    row = plan_model(SMALL, vram_mib=16000, ram_mib=32000)
    assert row["verdict"] == VERDICT_GPU
    assert row["gpu_layers"] == 28 and row["ctx_max"] == 4096


def test_fp16_7b_needs_cpu_offload():
    row = plan_model(FP16_7B, vram_mib=16000, ram_mib=32000)
    assert row["verdict"] == VERDICT_OFFLOAD
    assert 0 < row["gpu_layers"] < 28  # most layers on GPU, the rest on CPU
    assert row["ctx_gpu"] == 0  # weights leave no VRAM for KV
    assert row["ctx_max"] == 8192  # but GPU+RAM holds the full context


def test_too_big_for_vram_plus_ram_is_no():
    huge = {**SMALL, "params_b": 70, "quant": "q4_k_m"}
    row = plan_model(huge, vram_mib=16000, ram_mib=8000)
    assert row["verdict"] == VERDICT_NO


def test_target_context_beyond_budget_is_no():
    row = plan_model(
        {**SMALL, "max_context": 131072}, vram_mib=4000, ram_mib=3000, target_ctx=20000
    )
    assert row["verdict"] == VERDICT_NO and "exceeds" in row["note"]


def test_missing_arch_is_unknown_but_weight_feasible():
    row = plan_model(
        {"name": "m", "backend": "ollama", "params_b": 3.2, "quant": "q4_k_m"},
        vram_mib=16000,
        ram_mib=32000,
    )
    assert row["verdict"] == VERDICT_GPU  # weight-only feasibility still resolves
    assert "n_layers" in row["note"]


def test_no_spec_fields_is_unknown():
    row = plan_model({"name": "m", "backend": "ollama"}, vram_mib=16000, ram_mib=32000)
    assert row["verdict"] == VERDICT_UNKNOWN


# --- memory planner embedding-aware weight estimate -------------------------------------------------

# The Gemma-4 E4B w4a16 entry: weights MEASURED at 9.8 GiB on the RTX 4060 Ti (real-model validation). The
# int4 quant covers only linear layers; the embedding + Gemma 3n PLE stay bf16, captured by the
# measurement-anchored `hi_precision_params_b`.
E4B_W4A16 = {
    "name": "gemma-4-e4b-it-w4a16",
    "backend": "vllm",
    "params_b": 8,
    "quant": "w4a16",
    "n_layers": 42,
    "kv_dim": 512,
    "max_context": 131072,
    "vocab_size": 262144,
    "tie_word_embeddings": True,
    "hi_precision_params_b": 4.2,
}


def test_embedding_params_tied_vs_untied():
    assert embedding_params(1000, 10, tied=True) == 10000  # tied -> head shares the embedding
    assert embedding_params(1000, 10, tied=False) == 20000  # untied -> + a separate lm_head


def test_hi_precision_only_for_partial_quants():
    base = {"params_b": 12, "vocab_size": 262144, "hidden_size": 3840, "tie_word_embeddings": True}
    assert hi_precision_params({**base, "quant": "w4a16"}) > 0  # int4 keeps embedding bf16
    assert hi_precision_params({**base, "quant": "fp8"}) > 0  # fp8 keeps embedding bf16
    assert hi_precision_params({**base, "quant": "q4_k_m"}) == 0  # GGUF quantizes embedding too
    assert hi_precision_params({**base, "quant": "bf16"}) == 0  # uniform precision
    # an explicit override wins regardless of the quant family
    assert hi_precision_params({"quant": "bf16", "hi_precision_params_b": 4.2}) == 4.2e9


def test_weights_mib_detailed_prices_embedding_high():
    # 1B params, 4.5 bpw, with 0.5B held high-precision (16 bpw).
    flat = weights_mib(1.0, 4.5)
    detailed = weights_mib_detailed(1.0, 4.5, hi_params=0.5e9)
    assert detailed > flat
    # body: 0.5e9 * 4.5/8 ; hi: 0.5e9 * 16/8 ; total bytes / MiB
    assert round(detailed) == round((0.5e9 * 4.5 / 8 + 0.5e9 * 16 / 8) / (1024 * 1024))


def test_weights_mib_detailed_noop_for_full_precision():
    # bf16: high-precision floor == quant bpw, so the embedding is not magically cheaper.
    assert weights_mib_detailed(7.0, 16.0, hi_params=1e9) == weights_mib(7.0, 16.0)


def test_e4b_w4a16_weights_match_measured_floor():
    # The whole point of memory planner: the estimate lands on the MEASURED 9.8 GiB, not the flat ~4.2.
    row = plan_model(E4B_W4A16, vram_mib=16000, ram_mib=64000)
    assert row["weights_mib"] is not None
    gib = row["weights_mib"] / 1024
    assert 9.3 <= gib <= 10.3  # within ~0.5 GiB of the 9.8 GiB measured floor
    assert weights_mib(8, 4.5) / 1024 < 4.5  # the old flat estimate (the bug) was ~4.2 GiB


def test_w4a16_12b_embedding_premium():
    spec = {
        "name": "g12",
        "backend": "vllm",
        "params_b": 12,
        "quant": "w4a16",
        "vocab_size": 262144,
        "hidden_size": 3840,
        "tie_word_embeddings": True,
    }
    row = plan_model(spec, vram_mib=16000, ram_mib=64000)
    assert row["weights_mib"] / 1024 > weights_mib(12, 4.5) / 1024  # premium over flat
    assert round(row["weights_mib"] / 1024, 1) == 7.6  # 256k embedding priced at bf16


def test_arch_from_config_handles_nested_text_config():
    cfg = {
        "tie_word_embeddings": True,
        "text_config": {"vocab_size": 262144, "hidden_size": 3840, "num_hidden_layers": 48},
    }
    assert arch_from_config(cfg) == {
        "vocab_size": 262144,
        "hidden_size": 3840,
        "n_layers": 48,
        "tie_word_embeddings": True,
    }


def test_arch_from_config_flat_and_partial():
    out = arch_from_config({"vocab_size": 128256, "num_hidden_layers": 28})
    assert out == {"vocab_size": 128256, "n_layers": 28}  # missing fields simply omitted


def test_enrich_arch_fills_missing_from_cache(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "vocab_size": 262144,
                "hidden_size": 3840,
                "num_hidden_layers": 48,
                "tie_word_embeddings": True,
            }
        )
    )
    monkeypatch.setattr(planner, "cached_config_path", lambda _repo: cfg)
    spec = {"name": "g", "backend": "vllm", "source": "org/model", "params_b": 12, "quant": "w4a16"}
    out = enrich_arch(spec)
    assert out["vocab_size"] == 262144 and out["hidden_size"] == 3840
    assert out["n_layers"] == 48 and out["tie_word_embeddings"] is True
    # curated spec values win; the config only fills gaps
    assert enrich_arch({**spec, "hidden_size": 9999})["hidden_size"] == 9999


def test_enrich_arch_skips_non_hf_sources(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(planner, "cached_config_path", lambda r: seen.append(r))  # type: ignore[func-returns-value]
    ollama = enrich_arch({"name": "m", "backend": "ollama", "source": "llama3.2:3b"})
    gguf = enrich_arch({"name": "m", "backend": "ollama", "source": "hf.co/org/x-GGUF:Q4_K_M"})
    assert ollama.get("vocab_size") is None and gguf.get("vocab_size") is None
    assert seen == []  # the cache is never touched for non-"org/name" sources


# --- memory planner sliding-window KV + config override ---------------------------------------------


def test_attention_layer_split():
    assert attention_layer_split(34, 6) == (5, 29)  # Gemma3-style: 1 global per 6 layers
    assert attention_layer_split(48, 1) == (48, 0)  # pattern <= 1 -> all full attention
    assert attention_layer_split(48, 0) == (48, 0)


def test_kv_sliding_window_caps_past_the_window():
    # Past the window the sliding layers stop growing; full attention keeps growing linearly.
    full = kv_mib_at_context(34, 256, 8192)
    sliding = kv_mib_at_context(34, 256, 8192, sliding_window=1024, sliding_window_pattern=6)
    assert sliding < full  # sliding-window caches far less KV at a long context
    # below the window the two agree (every layer still grows)
    at_window = kv_mib_at_context(34, 256, 1024, sliding_window=1024, sliding_window_pattern=6)
    assert round(at_window, 4) == round(kv_mib_at_context(34, 256, 1024), 4)


def test_max_context_for_kv_sliding_allows_more_context():
    # Same VRAM budget admits a longer context once sliding-window KV is modeled.
    args = (12000.0, 9000.0, 512.0, 34, 256, 131072)
    full = max_context_for_kv(*args)
    sliding = max_context_for_kv(*args, sliding_window=1024, sliding_window_pattern=6)
    assert sliding > full
    # with no sliding fields it equals the linear full-attention max_context
    per_tok = kv_mib_per_token(34, 256)
    assert full == max_context(12000.0, 9000.0, 512.0, per_tok, 131072)


def test_arch_from_config_extracts_sliding_window():
    cfg = {"text_config": {"sliding_window": 1024, "sliding_window_pattern": 6}}
    out = arch_from_config(cfg)
    assert out["sliding_window"] == 1024 and out["sliding_window_pattern"] == 6


def test_arch_from_config_derives_pattern_from_layer_types():
    cfg = {
        "num_hidden_layers": 6,
        "sliding_window": 512,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    }
    assert arch_from_config(cfg)["sliding_window_pattern"] == 3  # 6 layers / 2 full = 3


def test_enrich_arch_override_replaces_curated(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"num_hidden_layers": 48, "sliding_window": 1024}))
    monkeypatch.setattr(planner, "cached_config_path", lambda _repo: cfg)
    spec = {"name": "g", "backend": "vllm", "source": "org/model", "n_layers": 99}
    # default fill-gaps keeps the curated (wrong) n_layers
    assert enrich_arch(spec)["n_layers"] == 99
    # override lets the real served config win
    out = enrich_arch(spec, override=True)
    assert out["n_layers"] == 48 and out["sliding_window"] == 1024


# --- Mistral Small 3.1 24B family default (gpu-tier-mistral-default) ---------------------------

# FP8 vLLM checkpoint: full attention (no sliding window), 131k vocab with an UNTIED lm_head that
# stays bf16, so the embedding-aware estimate must exceed the flat params_b x bpw.
MISTRAL_FP8 = {
    "name": "mistral-small-3.1-24b-fp8",
    "backend": "vllm",
    "params_b": 24,
    "quant": "fp8",
    "n_layers": 40,
    "kv_dim": 1024,
    "max_context": 131072,
    "vocab_size": 131072,
    "hidden_size": 5120,
    "tie_word_embeddings": False,
}
# The w4a16 vLLM checkpoint that fits GPU-resident on the 24 GiB tier (the fp8 ~24 GiB does not).
MISTRAL_W4A16 = {
    "name": "mistral-small-3.1-24b",
    "backend": "vllm",
    "params_b": 24,
    "quant": "w4a16",
    "n_layers": 40,
    "kv_dim": 1024,
    "max_context": 131072,
    "vocab_size": 131072,
    "hidden_size": 5120,
    "tie_word_embeddings": False,
}
# The q4_k_m GGUF that Ollama offloads on smaller tiers (k-quant prices the embedding too).
MISTRAL_GGUF = {
    "name": "mistral-small-3.1-24b-gguf",
    "backend": "ollama",
    "params_b": 24,
    "quant": "q4_k_m",
    "n_layers": 40,
    "kv_dim": 1024,
    "max_context": 131072,
}


def test_mistral_fp8_untied_embedding_premium_over_flat():
    # The untied 131k embedding stays bf16 under fp8, so the estimate must beat the flat product.
    row = plan_model(MISTRAL_FP8, vram_mib=32607, ram_mib=128 * 1024)
    flat = weights_mib(24, 8.0)
    assert row["weights_mib"] > flat
    assert 23.0 <= row["weights_mib"] / 1024 <= 24.5  # ~23.6 GiB priced weights


def test_mistral_fp8_needs_32gb_for_vllm_gpu_fit():
    # vLLM has no CPU offload, so the fp8 weights only hold a serving window on a 32 GiB card.
    assert plan_model(MISTRAL_FP8, vram_mib=16380, ram_mib=128 * 1024)["ctx_gpu"] == 0
    big = plan_model(MISTRAL_FP8, vram_mib=32607, ram_mib=128 * 1024)
    assert big["ctx_gpu"] >= 2048  # room for a real serving context once the weights fit VRAM


def test_mistral_gguf_offloads_on_16gb():
    # The q4_k_m GGUF (no embedding premium) runs on a 16 GiB card via GPU+RAM offload.
    row = plan_model(MISTRAL_GGUF, vram_mib=16380, ram_mib=128 * 1024)
    assert row["verdict"] == VERDICT_OFFLOAD
    assert row["ctx_max"] >= 2048
    assert row["weights_mib"] / 1024 < MISTRAL_FP8["params_b"] * 8 / 8 / 1.5  # q4 < fp8 weights


def test_mistral_w4a16_fits_gpu_resident_on_24gb():
    # gpu-tier-24-mistral-vllm: the w4a16 quant prices < 24 GiB and holds a real serving window
    # fully on a 24 GiB card (where the fp8 ~24 GiB weights leave no KV room).
    row = plan_model(MISTRAL_W4A16, vram_mib=24576, ram_mib=128 * 1024)
    assert row["weights_mib"] / 1024 < 24  # ~14.4 GiB priced weights
    assert row["weights_mib"] < plan_model(MISTRAL_FP8, 24576, 128 * 1024)["weights_mib"]  # < fp8
    assert row["ctx_gpu"] >= 2048  # a usable serving context fits VRAM at 24 GiB
    # ...and it does NOT clear a 16 GiB card's GPU window, so the resolver keeps offload there.
    assert plan_model(MISTRAL_W4A16, vram_mib=16380, ram_mib=128 * 1024)["ctx_gpu"] < 2048


def test_plan_model_sliding_window_fits_longer_context():
    base = {
        "name": "gemma",
        "backend": "vllm",
        "params_b": 12.0,
        "quant": "w4a16",
        "n_layers": 48,
        "kv_dim": 256,
        "max_context": 131072,
    }
    full = plan_model(base, vram_mib=12000, ram_mib=0)
    sliding = plan_model(
        {**base, "sliding_window": 1024, "sliding_window_pattern": 6}, vram_mib=12000, ram_mib=0
    )
    assert 0 < full["ctx_gpu"] < base["max_context"]  # full attention is KV-bound below the cap
    assert sliding["ctx_gpu"] > full["ctx_gpu"]  # sliding-window frees room for more context
