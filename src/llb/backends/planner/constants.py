"""Shared constants for host-feasibility planning."""

MIB = 1024 * 1024
KV_ELEM_BYTES = 2
EMBED_BPW = 16.0

QUANT_BPW = {
    "fp32": 32.0,
    "fp16": 16.0,
    "f16": 16.0,
    "bf16": 16.0,
    "fp8": 8.0,
    "q8_0": 8.5,
    "q6_k": 6.6,
    "q5_k_m": 5.5,
    "q5_0": 5.5,
    "q5_1": 5.6,
    "q4_k_m": 4.5,
    "q4_k_s": 4.3,
    "q4_0": 4.5,
    "q4_1": 4.8,
    "w4a16": 4.5,
    "int4": 4.5,
    "awq": 4.25,
    "gptq": 4.25,
    "q3_k_m": 3.9,
    "q3_k_s": 3.5,
    "iq3": 3.5,
    "q2_k": 3.0,
}

PARTIAL_QUANT_FORMATS = {"w4a16", "int4", "awq", "gptq", "fp8"}

DEFAULT_VRAM_RESERVE = 1024
DEFAULT_RAM_RESERVE = 2048
DEFAULT_OVERHEAD = 512
MIN_USABLE_CTX = 512

VERDICT_GPU = "gpu"
VERDICT_OFFLOAD = "offload"
VERDICT_NO = "no"
VERDICT_UNKNOWN = "unknown"
