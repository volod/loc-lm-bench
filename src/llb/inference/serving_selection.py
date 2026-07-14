"""Focused serving selection implementation."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from llb.backends.hardware import detect_gpus
from llb.core.paths import PROJECT_ROOT

SUPPORTED_TIERS_GB = (12, 16, 24, 32)

PRIMARY_TARGETS = ("mamaylm", "lapa", "gemma-4", "qwen3.6", "mistral")

GEMMA4_TARGET_PREFIX = "gemma-4"

DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "config-example" / "manifest.yaml"


@dataclass(frozen=True)
class GpuTierInfo:
    tier_gb: int
    total_mb: int
    gpu_name: str
    detected: bool


def bucket_vram_mb_to_tier(total_mb: int) -> int:
    """Map nvidia-smi total VRAM (MiB) to a supported tier (12/16/24/32 GiB).

    Thresholds use GiB with slack for cards that report less than nominal size
    (e.g. 16380 MiB -> 16 GiB tier).
    """
    gib = total_mb / 1024
    if gib < 14:
        return 12
    if gib < 20:
        return 16
    if gib < 28:
        return 24
    return 32


def detect_gpu_tier() -> GpuTierInfo:
    """Detect the primary GPU and return its serving tier."""
    gpus = detect_gpus()
    if not gpus:
        return GpuTierInfo(tier_gb=16, total_mb=0, gpu_name="", detected=False)
    primary = max(gpus, key=lambda g: g.total_mb)
    tier = bucket_vram_mb_to_tier(primary.total_mb)
    return GpuTierInfo(
        tier_gb=tier,
        total_mb=primary.total_mb,
        gpu_name=primary.name,
        detected=True,
    )


def resolve_tier(gpu_gb: int | None) -> GpuTierInfo:
    if gpu_gb is not None:
        if gpu_gb not in SUPPORTED_TIERS_GB:
            raise ValueError(
                f"unsupported GPU tier {gpu_gb} GiB; choose one of {list(SUPPORTED_TIERS_GB)}"
            )
        detected = detect_gpu_tier()
        return GpuTierInfo(
            tier_gb=gpu_gb,
            total_mb=detected.total_mb,
            gpu_name=detected.gpu_name,
            detected=detected.detected,
        )
    return detect_gpu_tier()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or DEFAULT_MANIFEST
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest must be a mapping: {manifest_path}")
    return raw


def _tier_entries(manifest: dict[str, Any], tier_gb: int) -> dict[str, Any]:
    tiers = manifest.get("tiers")
    if not isinstance(tiers, dict):
        raise ValueError("manifest.tiers must be a mapping")
    entry = tiers.get(tier_gb)
    if entry is None:
        entry = tiers.get(str(tier_gb))
    if not isinstance(entry, dict):
        raise ValueError(f"manifest has no tier entry for {tier_gb} GiB GPU")
    return entry


def _model_size_b(target_id: str, model: str) -> float:
    text = f"{target_id} {model}".lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*b", text)
    return max((float(value) for value in matches), default=0.0)


def _gemma4_rank(row: dict[str, Any], allow_cuda: bool) -> tuple[int, float]:
    backend = str(row["backend"])
    cuda_score = 1 if allow_cuda and backend == "vllm" else 0
    return cuda_score, _model_size_b(str(row["target"]), str(row["model"]))


def _supports_min_context(row: dict[str, Any], min_context_tokens: int | None) -> bool:
    """Whether a serving row has enough configured context for the requested workflow."""
    if not min_context_tokens:
        return True
    if row.get("backend") != "vllm":
        return True
    max_model_len = row.get("max_model_len")
    return isinstance(max_model_len, int) and max_model_len >= min_context_tokens


def select_host_gemma4_target(
    *,
    gpu_gb: int | None = None,
    manifest_path: Path | None = None,
    min_context_tokens: int | None = None,
) -> dict[str, Any]:
    """Return the most capable Gemma 4 target for the resolved CUDA tier.

    CUDA hosts prefer vLLM Gemma 4 rows over larger Ollama/offload rows. Within the same backend
    class, larger Gemma 4 parameter counts win. CPU/no-GPU fallback still returns a Gemma 4 row,
    but it does not prefer vLLM unless the caller explicitly supplied a GPU tier. When
    `min_context_tokens` is set, short-context vLLM eval cells are ignored for long-prompt
    workflows such as corpus drafting.
    """
    tier_info = resolve_tier(gpu_gb)
    manifest = load_manifest(manifest_path)
    entries = _tier_entries(manifest, tier_info.tier_gb)
    allow_cuda = tier_info.detected or gpu_gb is not None
    rows = [
        row
        for target_id, entry in entries.items()
        if _is_gemma4_entry(target_id, entry)
        for row in [_gemma4_row(tier_info, target_id, entry)]
        if _supports_min_context(row, min_context_tokens)
    ]
    if not rows:
        suffix = f" with context >= {min_context_tokens} tokens" if min_context_tokens else ""
        raise ValueError(f"tier {tier_info.tier_gb}: no Gemma 4 serving target{suffix}")
    return max(rows, key=lambda row: _gemma4_rank(row, allow_cuda))


def _is_gemma4_entry(target_id: str, entry: Any) -> bool:
    """True for well-formed manifest entries under the Gemma 4 target family."""
    if not isinstance(entry, dict):
        return False
    return target_id == GEMMA4_TARGET_PREFIX or target_id.startswith(f"{GEMMA4_TARGET_PREFIX}-")


_GEMMA4_OPTIONAL_FIELDS: tuple[tuple[str, Any], ...] = (
    ("gpu_memory_utilization", float),
    ("max_model_len", int),
    ("cpu_offload_gb", float),
    ("kv_offloading_size_gb", float),
)


def _gemma4_row(tier_info: Any, target_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """One candidate serving row: tier context + the entry's backend/model + optional knobs."""
    row: dict[str, Any] = {
        "tier_gb": tier_info.tier_gb,
        "gpu_total_mb": tier_info.total_mb,
        "gpu_name": tier_info.gpu_name,
        "gpu_detected": tier_info.detected,
        "target": target_id,
        "backend": str(entry["backend"]),
        "model": str(entry["model"]),
    }
    for field, cast_fn in _GEMMA4_OPTIONAL_FIELDS:
        if entry.get(field) is not None:
            row[field] = cast_fn(entry[field])
    return row


def format_detect_line(info: GpuTierInfo) -> str:
    if info.detected:
        return (
            f"gpu_tier={info.tier_gb} total_mb={info.total_mb} "
            f"name={info.gpu_name!r} supported={list(SUPPORTED_TIERS_GB)}"
        )
    return (
        f"gpu_tier={info.tier_gb} total_mb=0 name= supported={list(SUPPORTED_TIERS_GB)} "
        "(no GPU detected)"
    )
