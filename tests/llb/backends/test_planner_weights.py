"""Tests for planner weights."""

from llb.backends.planner.plan import plan_model
from llb.backends.planner.weights import (
    embedding_params,
    hi_precision_params,
    weights_mib,
    weights_mib_detailed,
)
from test_planner import E4B_W4A16


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
