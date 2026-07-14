"""GPU-tier-adaptive Qwen/Gemma model selection for local draft comparisons."""

from dataclasses import asdict, dataclass

from llb.inference.serving_selection import GpuTierInfo, detect_gpu_tier
from llb.prep.ontology.ollama_lifecycle import ollama_native_root


@dataclass(frozen=True)
class LocalCompareProfile:
    tier_gb: int
    baseline_model: str
    probe_model: str
    num_ctx: int


LOCAL_COMPARE_PROFILES = {
    12: LocalCompareProfile(12, "qwen3:8b", "gemma4:e2b", 8192),
    16: LocalCompareProfile(16, "qwen3:14b", "gemma4:e4b", 8192),
    24: LocalCompareProfile(24, "qwen3:30b", "gemma4:26b", 8192),
    32: LocalCompareProfile(32, "qwen3:30b", "gemma4:31b", 16384),
}


def installed_ollama_models(base_url: str) -> set[str]:
    import httpx

    response = httpx.get(f"{ollama_native_root(base_url)}/api/tags", timeout=10.0)
    response.raise_for_status()
    return {
        str(entry.get("name") or entry.get("model"))
        for entry in response.json().get("models", [])
        if entry.get("name") or entry.get("model")
    }


def select_local_compare_models(
    base_url: str,
    *,
    baseline_model: str | None = None,
    probe_model: str | None = None,
    gpu: GpuTierInfo | None = None,
    installed: set[str] | None = None,
) -> tuple[str, str, int, dict[str, object]]:
    """Resolve a fitting Qwen/Gemma pair and require both tags to exist locally."""
    detected = gpu or detect_gpu_tier()
    profile = LOCAL_COMPARE_PROFILES[detected.tier_gb]
    baseline = baseline_model or profile.baseline_model
    probe = probe_model or profile.probe_model
    available = installed if installed is not None else installed_ollama_models(base_url)
    missing = [model for model in (baseline, probe) if model not in available]
    if missing:
        pulls = " ".join(f"ollama pull {model}" for model in missing)
        raise RuntimeError(
            f"selected local comparison model(s) are not installed: {missing}; {pulls}"
        )
    selection: dict[str, object] = {
        "policy": "gpu-tier-qwen-gemma",
        "detected": asdict(detected),
        "profile": asdict(profile),
        "baseline_source": "override" if baseline_model else "profile",
        "probe_source": "override" if probe_model else "profile",
    }
    return baseline, probe, profile.num_ctx, selection
