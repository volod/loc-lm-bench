from llb.backends.hardware import parse_meminfo
from llb.backends.planner.constants import (
    VERDICT_GPU,
    VERDICT_NO,
    VERDICT_OFFLOAD,
    VERDICT_UNKNOWN,
)
from llb.backends.planner.kv import (
    kv_mib_per_token,
    max_context,
)
from llb.backends.planner.plan import plan_model
from llb.backends.planner.weights import (
    resolve_bpw,
    weights_mib,
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


# --- memory planner sliding-window KV + config override ---------------------------------------------


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
