"""Feasibility planner: can a model be benchmarked on THIS host, and at what context.

We optimize ABILITY TO RUN, not speed: the memory budget is GPU VRAM + system RAM, and a
model that does not fit in VRAM alone may still run by splitting its layers between GPU and
CPU (llama.cpp `n_gpu_layers`, single sequence -- no batching, no parallel requests). The
available context size is whatever the combined budget allows after model weights, since the
KV cache grows linearly with context (batch=1).

Memory model (estimates, MiB):
  weights      = params_b * 1e9 * bits_per_weight / 8
  kv per token = 2 (K+V) * n_layers * kv_dim * 2 bytes (fp16 KV)   # batch 1, no parallel
  footprint(ctx) = weights + kv_per_token * ctx + compute_overhead

Budgets:
  VRAM usable = total_vram - vram_reserve   (CUDA runtime + display headroom)
  RAM  usable = total_ram  - ram_reserve    (OS headroom)
  total       = VRAM usable + RAM usable

Per model we report: max context fully on GPU (`ctx_gpu`), max context using GPU+RAM offload
(`ctx_max`), the GPU/CPU layer split at the planning context, and a verdict
(gpu / offload / no / unknown). Everything is a planning estimate; the real fit test is a
launch (Milestone 2).
"""

MIB = 1024 * 1024
KV_ELEM_BYTES = 2  # fp16 KV cache element

# Bits-per-weight for common quantizations (GGUF k-quants + plain dtypes).
QUANT_BPW = {
    "fp16": 16.0, "f16": 16.0, "bf16": 16.0, "fp32": 32.0,
    "q8_0": 8.5, "q6_k": 6.6, "q5_k_m": 5.5, "q5_0": 5.5, "q5_1": 5.6,
    "q4_k_m": 4.5, "q4_k_s": 4.3, "q4_0": 4.5, "q4_1": 4.8,
    "q3_k_m": 3.9, "q3_k_s": 3.5, "q2_k": 3.0,
}

# Defaults (MiB) -- conservative headroom so the plan is honest, not optimistic.
DEFAULT_VRAM_RESERVE = 1024
DEFAULT_RAM_RESERVE = 2048
DEFAULT_OVERHEAD = 512
MIN_USABLE_CTX = 512

VERDICT_GPU = "gpu"           # fits fully in VRAM at the planning context
VERDICT_OFFLOAD = "offload"   # runs only by splitting layers to CPU RAM
VERDICT_NO = "no"             # does not fit even in VRAM + RAM
VERDICT_UNKNOWN = "unknown"   # missing the spec fields needed to plan


def resolve_bpw(spec: dict) -> float | None:
    if spec.get("bpw") is not None:
        return float(spec["bpw"])
    quant = str(spec.get("quant", "")).lower()
    return QUANT_BPW.get(quant)


def weights_mib(params_b: float, bpw: float) -> float:
    return params_b * 1e9 * bpw / 8 / MIB


def kv_mib_per_token(n_layers: int, kv_dim: int) -> float:
    """MiB of KV cache per token (K and V, all layers, batch 1)."""
    return 2 * n_layers * kv_dim * KV_ELEM_BYTES / MIB


def max_context(budget_mib: float, w_mib: float, overhead_mib: float,
                per_tok_mib: float, cap: int) -> int:
    """Largest context (<= cap) whose footprint fits in `budget_mib`. 0 if weights don't fit."""
    avail = budget_mib - w_mib - overhead_mib
    if avail <= 0 or per_tok_mib <= 0:
        return 0
    return max(0, min(cap, int(avail / per_tok_mib)))


def gpu_layers(vram_usable_mib: float, overhead_mib: float, w_mib: float,
               kv_at_ctx_mib: float, n_layers: int) -> int:
    """How many of `n_layers` (weights + their KV) fit in VRAM; the rest go to CPU RAM."""
    per_layer = (w_mib + kv_at_ctx_mib) / n_layers
    if per_layer <= 0:
        return n_layers
    fit = int((vram_usable_mib - overhead_mib) / per_layer)
    return max(0, min(n_layers, fit))


def plan_model(
    spec: dict,
    vram_mib: int,
    ram_mib: int,
    *,
    vram_reserve: int = DEFAULT_VRAM_RESERVE,
    ram_reserve: int = DEFAULT_RAM_RESERVE,
    overhead_mib: int = DEFAULT_OVERHEAD,
    target_ctx: int | None = None,
    min_ctx: int = MIN_USABLE_CTX,
) -> dict:
    """Plan one model on the host. Returns a row dict (see module docstring)."""
    row = {
        "name": spec.get("name", spec.get("source", "?")),
        "backend": spec.get("backend", "?"),
        "params_b": spec.get("params_b"),
        "quant": spec.get("quant") or (f"{spec['bpw']}bpw" if spec.get("bpw") else None),
        "weights_mib": None, "n_layers": spec.get("n_layers"),
        "ctx_gpu": 0, "ctx_max": 0, "gpu_layers": 0,
        "verdict": VERDICT_UNKNOWN, "note": "",
    }

    bpw = resolve_bpw(spec)
    params_b = spec.get("params_b")
    if bpw is None or params_b is None:
        row["note"] = "add params_b + quant/bpw to plan"
        return row
    w = weights_mib(float(params_b), bpw)
    row["weights_mib"] = w

    vram_usable = max(0, vram_mib - vram_reserve)
    ram_usable = max(0, ram_mib - ram_reserve)
    total = vram_usable + ram_usable

    n_layers = spec.get("n_layers")
    kv_dim = spec.get("kv_dim")
    cap = spec.get("max_context")
    if not (n_layers and kv_dim and cap):
        # Weight-only feasibility (no architecture -> cannot size the KV cache).
        row["note"] = "add n_layers + kv_dim + max_context for context planning"
        if w + overhead_mib <= vram_usable:
            row["verdict"] = VERDICT_GPU
        elif w + overhead_mib <= total:
            row["verdict"] = VERDICT_OFFLOAD
        else:
            row["verdict"] = VERDICT_NO
        return row

    per_tok = kv_mib_per_token(n_layers, kv_dim)
    row["ctx_gpu"] = max_context(vram_usable, w, overhead_mib, per_tok, cap)
    row["ctx_max"] = max_context(total, w, overhead_mib, per_tok, cap)

    if target_ctx is not None and target_ctx > row["ctx_max"]:
        row["verdict"] = VERDICT_NO
        row["note"] = f"context {target_ctx} exceeds the {row['ctx_max']} the host can hold"
        return row
    if row["ctx_max"] < min_ctx:
        row["verdict"] = VERDICT_NO
        row["note"] = "weights leave no room for a usable KV cache"
        return row

    planning_ctx = target_ctx if target_ctx is not None else row["ctx_max"]
    kv_at = per_tok * planning_ctx
    gl = gpu_layers(vram_usable, overhead_mib, w, kv_at, n_layers)
    row["gpu_layers"] = gl
    row["verdict"] = VERDICT_GPU if gl >= n_layers else VERDICT_OFFLOAD
    row["note"] = f"plan @ ctx={planning_ctx}"
    return row


def plan_models(models: list[dict], vram_mib: int, ram_mib: int, **kwargs) -> list[dict]:
    return [plan_model(m, vram_mib, ram_mib, **kwargs) for m in models]


def _gb(mib) -> str:
    return "-" if mib is None else f"{mib / 1024:.1f}"


def format_plan(rows: list[dict], vram_mib: int, ram_mib: int) -> str:
    """ASCII table of the plan."""
    headers = ["model", "backend", "params", "quant", "wt_GB",
               "ctx_gpu", "ctx_max", "gpu/total", "verdict"]

    def fmt(row: dict) -> list[str]:
        n_layers = row["n_layers"]
        split = "-" if not n_layers else f"{row['gpu_layers']}/{n_layers}"
        return [
            row["name"], row["backend"],
            "-" if row["params_b"] is None else f"{row['params_b']}B",
            row["quant"] or "-", _gb(row["weights_mib"]),
            str(row["ctx_gpu"]) if row["ctx_gpu"] else "-",
            str(row["ctx_max"]) if row["ctx_max"] else "-",
            split, row["verdict"],
        ]

    table = [fmt(r) for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in table)) if table else len(h)
              for i, h in enumerate(headers)]
    out = [
        f"host budget: VRAM {vram_mib} MiB + RAM {ram_mib} MiB "
        f"(usable after reserves; weights + KV must fit the combined budget)",
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for r in table:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)
