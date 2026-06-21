"""Inference serving config generation."""

from llb.inference.generate import (
    DEFAULT_MANIFEST,
    SUPPORTED_TIERS_GB,
    GpuTierInfo,
    bucket_vram_mb_to_tier,
    detect_gpu_tier,
    format_detect_line,
    generate_serving_configs,
    resolve_tier,
)

__all__ = [
    "DEFAULT_MANIFEST",
    "SUPPORTED_TIERS_GB",
    "GpuTierInfo",
    "bucket_vram_mb_to_tier",
    "detect_gpu_tier",
    "format_detect_line",
    "generate_serving_configs",
    "resolve_tier",
]
