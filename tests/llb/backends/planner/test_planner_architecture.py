"""Tests for planner architecture."""

import json
import llb.backends.planner.architecture as architecture
from llb.backends.planner.architecture import arch_from_config, enrich_arch


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
    monkeypatch.setattr(architecture, "cached_config_path", lambda _repo: cfg)
    spec = {"name": "g", "backend": "vllm", "source": "org/model", "params_b": 12, "quant": "w4a16"}
    out = enrich_arch(spec)
    assert out["vocab_size"] == 262144 and out["hidden_size"] == 3840
    assert out["n_layers"] == 48 and out["tie_word_embeddings"] is True
    # curated spec values win; the config only fills gaps
    assert enrich_arch({**spec, "hidden_size": 9999})["hidden_size"] == 9999


def test_enrich_arch_still_fills_optional_hybrid_fields(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "layer_types": ["linear_attention", "full_attention"] * 4,
                "num_hidden_layers": 8,
            }
        )
    )
    monkeypatch.setattr(architecture, "cached_config_path", lambda _repo: cfg)
    spec = {
        "name": "hybrid",
        "backend": "vllm",
        "source": "org/model",
        "vocab_size": 10,
        "hidden_size": 16,
        "n_layers": 8,
        "kv_dim": 64,
        "max_context": 8192,
        "tie_word_embeddings": True,
    }

    assert enrich_arch(spec)["kv_layers"] == 4


def test_arch_from_config_extracts_context_kv_and_hybrid_attention_layers():
    layer_types = ["linear_attention", "linear_attention", "full_attention"] * 4
    out = arch_from_config(
        {
            "text_config": {
                "num_hidden_layers": 12,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "max_position_embeddings": 262144,
                "layer_types": layer_types,
            }
        }
    )

    assert out["kv_dim"] == 1024
    assert out["kv_layers"] == 4
    assert out["max_context"] == 262144


def test_enrich_arch_skips_non_hf_sources(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(architecture, "cached_config_path", lambda r: seen.append(r))  # type: ignore[func-returns-value]
    ollama = enrich_arch({"name": "m", "backend": "ollama", "source": "llama3.2:3b"})
    gguf = enrich_arch({"name": "m", "backend": "ollama", "source": "hf.co/org/x-GGUF:Q4_K_M"})
    assert ollama.get("vocab_size") is None and gguf.get("vocab_size") is None
    assert seen == []  # the cache is never touched for non-"org/name" sources
