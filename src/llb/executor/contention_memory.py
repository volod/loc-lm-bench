"""Focused contention memory implementation."""

from pathlib import Path
from typing import Any, cast

DEFAULT_VLLM_OVERHEAD_MB = 2048

DEFAULT_MIN_KV_HEADROOM_MB = 512

DEFAULT_MIN_SERVING_CTX = 2048

DEFAULT_MANIFEST = Path("samples/configs/models_uk.yaml")


def _spec_for(model_source: str, manifest: Path) -> "dict[str, Any] | None":
    from llb.backends.prepare.manifest import load_manifest

    for spec in load_manifest(manifest):
        if spec.get("source") == model_source or spec.get("name") == model_source:
            return cast("dict[str, Any]", spec)
    return None


def model_weight_floor_mb(model_source: str, manifest: Path = DEFAULT_MANIFEST) -> float:
    """Embedding-aware weights estimate (MiB, memory planner) for a model by source/name. 0.0 if unknown."""
    try:
        from llb.backends.planner.architecture import enrich_arch
        from llb.backends.planner.plan import plan_model

        spec = _spec_for(model_source, manifest)
        if spec is None:
            return 0.0
        row = plan_model(enrich_arch(cast(Any, spec)), vram_mib=1_000_000, ram_mib=1_000_000)
        return float(row["weights_mib"] or 0.0)
    except Exception:
        return 0.0


def model_kv_headroom_mb(
    model_source: str,
    manifest: Path = DEFAULT_MANIFEST,
    *,
    min_serving_ctx: int = DEFAULT_MIN_SERVING_CTX,
) -> int:
    """Arch-derived minimal KV working set (MiB, VRAM contention guard): the KV the model needs to serve at least
    `min_serving_ctx` tokens, sliding-window-aware. The abort check uses this instead of a fixed
    floor, so a model with a heavy KV per token is judged un-launchable at the right threshold.
    Falls back to the fixed floor when the arch is unknown."""
    try:
        from llb.backends.planner.architecture import enrich_arch
        from llb.backends.planner.kv import kv_mib_at_context

        spec = _spec_for(model_source, manifest)
        if spec is None:
            return DEFAULT_MIN_KV_HEADROOM_MB
        enriched = cast("dict[str, Any]", enrich_arch(cast(Any, spec)))
        n_layers = enriched.get("n_layers")
        kv_dim = enriched.get("kv_dim")
        if not (n_layers and kv_dim):
            return DEFAULT_MIN_KV_HEADROOM_MB
        kv = kv_mib_at_context(
            int(n_layers),
            int(kv_dim),
            min_serving_ctx,
            sliding_window=enriched.get("sliding_window"),
            sliding_window_pattern=enriched.get("sliding_window_pattern"),
        )
        return max(DEFAULT_MIN_KV_HEADROOM_MB, int(kv))
    except Exception:
        return DEFAULT_MIN_KV_HEADROOM_MB
