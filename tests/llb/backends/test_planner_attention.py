"""Tests for planner attention."""

import json
import llb.backends.planner.architecture as architecture
from llb.backends.planner.architecture import arch_from_config, enrich_arch
from llb.backends.planner.constants import (
    VERDICT_OFFLOAD,
)
from llb.backends.planner.kv import (
    attention_layer_split,
    kv_mib_at_context,
    kv_mib_per_token,
    max_context,
    max_context_for_kv,
)
from llb.backends.planner.plan import plan_model
from llb.backends.planner.weights import (
    weights_mib,
)
from test_planner import MISTRAL_FP8, MISTRAL_GGUF, MISTRAL_W4A16


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
    monkeypatch.setattr(architecture, "cached_config_path", lambda _repo: cfg)
    spec = {"name": "g", "backend": "vllm", "source": "org/model", "n_layers": 99}
    # default fill-gaps keeps the curated (wrong) n_layers
    assert enrich_arch(spec)["n_layers"] == 99
    # override lets the real served config win
    out = enrich_arch(spec, override=True)
    assert out["n_layers"] == 48 and out["sliding_window"] == 1024


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


def test_plan_model_hybrid_attention_only_prices_kv_bearing_layers():
    base = {
        "name": "hybrid",
        "backend": "vllm",
        "params_b": 8.0,
        "quant": "q4_k_m",
        "n_layers": 64,
        "kv_dim": 1024,
        "max_context": 262144,
    }

    all_attention = plan_model(base, vram_mib=16000, ram_mib=0)
    hybrid = plan_model({**base, "kv_layers": 16}, vram_mib=16000, ram_mib=0)

    assert hybrid["ctx_gpu"] > all_attention["ctx_gpu"]
    assert hybrid["gpu_layers"] <= base["n_layers"]
