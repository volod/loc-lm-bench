import json

import llb.backends.planner as planner
from llb.backends.hardware import parse_meminfo
from llb.backends.planner import (
    VERDICT_GPU,
    VERDICT_NO,
    VERDICT_OFFLOAD,
    VERDICT_UNKNOWN,
    arch_from_config,
    embedding_params,
    enrich_arch,
    hi_precision_params,
    kv_mib_per_token,
    max_context,
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


# --- M4.1 embedding-aware weight estimate -------------------------------------------------

# The Gemma-4 E4B w4a16 entry: weights MEASURED at 9.8 GiB on the RTX 4060 Ti (M2.4). The
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
    # The whole point of M4.1: the estimate lands on the MEASURED 9.8 GiB, not the flat ~4.2.
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
