"""KV-cache sizing and GPU-layer allocation."""

from llb.backends.planner.constants import KV_ELEM_BYTES, MIB


def kv_mib_per_token(n_layers: int, kv_dim: int) -> float:
    """MiB of K and V cache per token for all layers at batch size one."""
    return 2 * n_layers * kv_dim * KV_ELEM_BYTES / MIB


def attention_layer_split(n_layers: int, sliding_window_pattern: int) -> tuple[int, int]:
    """Return full-attention and sliding-attention layer counts."""
    if sliding_window_pattern <= 1:
        return n_layers, 0
    full = max(1, n_layers // sliding_window_pattern)
    return full, max(0, n_layers - full)


def kv_mib_at_context(
    n_layers: int,
    kv_dim: int,
    ctx: int,
    *,
    sliding_window: int | None = None,
    sliding_window_pattern: int | None = None,
) -> float:
    """Calculate KV MiB at a context length, including sliding-window layer caps."""
    per_tok_layer = kv_mib_per_token(1, kv_dim)
    if not sliding_window or not sliding_window_pattern or sliding_window_pattern <= 1:
        return per_tok_layer * n_layers * ctx
    full, sliding = attention_layer_split(n_layers, sliding_window_pattern)
    return per_tok_layer * full * ctx + per_tok_layer * sliding * min(ctx, sliding_window)


def max_context(
    budget_mib: float, w_mib: float, overhead_mib: float, per_tok_mib: float, cap: int
) -> int:
    """Largest context within a linear KV budget."""
    avail = budget_mib - w_mib - overhead_mib
    if avail <= 0 or per_tok_mib <= 0:
        return 0
    return max(0, min(cap, int(avail / per_tok_mib)))


def max_context_for_kv(
    budget_mib: float,
    w_mib: float,
    overhead_mib: float,
    n_layers: int,
    kv_dim: int,
    cap: int,
    *,
    sliding_window: int | None = None,
    sliding_window_pattern: int | None = None,
) -> int:
    """Largest context within a full- or sliding-attention KV budget."""
    avail = budget_mib - w_mib - overhead_mib
    if avail <= 0:
        return 0
    per_tok_layer = kv_mib_per_token(1, kv_dim)
    if per_tok_layer <= 0:
        return 0
    if not sliding_window or not sliding_window_pattern or sliding_window_pattern <= 1:
        return max(0, min(cap, int(avail / (per_tok_layer * n_layers))))
    full, sliding = attention_layer_split(n_layers, sliding_window_pattern)
    linear_ctx = int(avail / (per_tok_layer * n_layers))
    if linear_ctx <= sliding_window:
        return max(0, min(cap, linear_ctx))
    after_sliding = avail - per_tok_layer * sliding * sliding_window
    if full == 0 or after_sliding <= 0:
        return max(0, min(cap, sliding_window))
    return max(0, min(cap, int(after_sliding / (per_tok_layer * full))))


def gpu_layers(
    vram_usable_mib: float, overhead_mib: float, w_mib: float, kv_at_ctx_mib: float, n_layers: int
) -> int:
    """Number of layers whose weights and KV fit in usable VRAM."""
    per_layer = (w_mib + kv_at_ctx_mib) / n_layers
    if per_layer <= 0:
        return n_layers
    fit = int((vram_usable_mib - overhead_mib) / per_layer)
    return max(0, min(n_layers, fit))
