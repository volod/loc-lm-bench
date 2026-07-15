"""Host-feasibility planning for one model or a model roster."""

from typing import Any

from llb.backends.planner.constants import (
    DEFAULT_OVERHEAD,
    DEFAULT_RAM_RESERVE,
    DEFAULT_VRAM_RESERVE,
    EMBED_BPW,
    MIN_USABLE_CTX,
    VERDICT_GPU,
    VERDICT_NO,
    VERDICT_OFFLOAD,
    VERDICT_UNKNOWN,
)
from llb.backends.planner.kv import gpu_layers, kv_mib_at_context, max_context_for_kv
from llb.backends.planner.weights import hi_precision_params, resolve_bpw, weights_mib_detailed
from llb.core.contracts.models import ModelPlanRow, ModelSpec


def _base_row(spec: ModelSpec) -> ModelPlanRow:
    return {
        "name": spec.get("name", spec.get("source", "?")),
        "backend": spec.get("backend", "?"),
        "params_b": spec.get("params_b"),
        "quant": spec.get("quant") or (f"{spec['bpw']}bpw" if spec.get("bpw") else None),
        "weights_mib": None,
        "n_layers": spec.get("n_layers"),
        "ctx_gpu": 0,
        "ctx_max": 0,
        "gpu_layers": 0,
        "verdict": VERDICT_UNKNOWN,
        "note": "",
    }


def _weight_only_verdict(
    row: ModelPlanRow, weights: float, overhead_mib: int, vram_usable: int, total: int
) -> ModelPlanRow:
    row["note"] = "add n_layers + kv_dim + max_context for context planning"
    if weights + overhead_mib <= vram_usable:
        row["verdict"] = VERDICT_GPU
    elif weights + overhead_mib <= total:
        row["verdict"] = VERDICT_OFFLOAD
    else:
        row["verdict"] = VERDICT_NO
    return row


def plan_model(
    spec: ModelSpec,
    vram_mib: int,
    ram_mib: int,
    *,
    vram_reserve: int = DEFAULT_VRAM_RESERVE,
    ram_reserve: int = DEFAULT_RAM_RESERVE,
    overhead_mib: int = DEFAULT_OVERHEAD,
    target_ctx: int | None = None,
    min_ctx: int = MIN_USABLE_CTX,
) -> ModelPlanRow:
    """Plan a model's weight, context, and GPU/CPU-layer fit on one host."""
    row = _base_row(spec)
    bpw = resolve_bpw(spec)
    params_b = spec.get("params_b")
    if bpw is None or params_b is None:
        row["note"] = "add params_b + quant/bpw to plan"
        return row

    weights = weights_mib_detailed(
        float(params_b),
        bpw,
        hi_precision_params(spec),
        float(spec.get("embed_bpw") or EMBED_BPW),
    )
    row["weights_mib"] = weights
    vram_usable = max(0, vram_mib - vram_reserve)
    total = vram_usable + max(0, ram_mib - ram_reserve)

    n_layers = spec.get("n_layers")
    kv_dim = spec.get("kv_dim")
    cap = spec.get("max_context")
    if not (n_layers and kv_dim and cap):
        return _weight_only_verdict(row, weights, overhead_mib, vram_usable, total)

    kv_kwargs = {
        "sliding_window": spec.get("sliding_window"),
        "sliding_window_pattern": spec.get("sliding_window_pattern"),
    }
    row["ctx_gpu"] = max_context_for_kv(
        vram_usable, weights, overhead_mib, n_layers, kv_dim, cap, **kv_kwargs
    )
    row["ctx_max"] = max_context_for_kv(
        total, weights, overhead_mib, n_layers, kv_dim, cap, **kv_kwargs
    )
    if target_ctx is not None and target_ctx > row["ctx_max"]:
        row["verdict"] = VERDICT_NO
        row["note"] = f"context {target_ctx} exceeds the {row['ctx_max']} the host can hold"
        return row
    if row["ctx_max"] < min_ctx:
        row["verdict"] = VERDICT_NO
        row["note"] = "weights leave no room for a usable KV cache"
        return row

    planning_ctx = target_ctx if target_ctx is not None else row["ctx_max"]
    kv_at = kv_mib_at_context(n_layers, kv_dim, planning_ctx, **kv_kwargs)
    row["gpu_layers"] = gpu_layers(vram_usable, overhead_mib, weights, kv_at, n_layers)
    row["verdict"] = VERDICT_GPU if row["gpu_layers"] >= n_layers else VERDICT_OFFLOAD
    row["note"] = f"plan @ ctx={planning_ctx}"
    return row


def plan_models(
    models: list[ModelSpec], vram_mib: int, ram_mib: int, **kwargs: Any
) -> list[ModelPlanRow]:
    return [plan_model(model, vram_mib, ram_mib, **kwargs) for model in models]
