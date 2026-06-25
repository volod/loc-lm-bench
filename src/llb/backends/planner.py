"""Feasibility planner: can a model be benchmarked on THIS host, and at what context.

We optimize ABILITY TO RUN, not speed: the memory budget is GPU VRAM + system RAM, and a
model that does not fit in VRAM alone may still run by splitting its layers between GPU and
CPU (llama.cpp `n_gpu_layers`, single sequence -- no batching, no parallel requests). The
available context size is whatever the combined budget allows after model weights, since the
KV cache grows linearly with context (batch=1).

Memory model (estimates, MiB):
  weights      = hi_precision_params * embed_bpw/8 + quantized_params * quant_bpw/8
  kv per token = 2 (K+V) * n_layers * kv_dim * 2 bytes (fp16 KV)   # batch 1, no parallel
  footprint(ctx) = weights + kv(ctx) + compute_overhead

KV is SLIDING-WINDOW-AWARE (M4.1): Gemma 3/4 interleave sliding-window layers (KV pinned at the
`sliding_window` size) with a periodic full-attention layer (`sliding_window_pattern`), so past the
window only the full-attention layers keep growing -- which is why a 12B Gemma fits a 16 GB card at
a long context. Absent those fields the estimate is the linear full-attention `kv_per_token * ctx`.
Arch fields come from the spec or a cached `config.json` (`enrich_arch`, which can also OVERRIDE
curated guesses with the real served config).

Weights are EMBEDDING-AWARE (M4.1): partial quants (w4a16 / int4 / fp8) quantize only the
linear layers, while the token embedding + norms stay high-precision. With a 256k-token vocab
that premium is large -- measured gemma-4-E4B w4a16 loads 9.8 GiB, not the 4.2 GiB a flat
`params_b x bpw` predicts. We price the high-precision mass (vocab embedding, untied output
head, or an explicit `hi_precision_params_b` for quirks like Gemma 3n Per-Layer Embeddings) at
`embed_bpw` and only the remainder at the quant bpw. Arch fields come from the spec or a cached
`config.json` (`enrich_arch`); when none are present the estimate falls back to the flat product.

Budgets:
  VRAM usable = total_vram - vram_reserve   (CUDA runtime + display headroom)
  RAM  usable = total_ram  - ram_reserve    (OS headroom)
  total       = VRAM usable + RAM usable

Per model we report: max context fully on GPU (`ctx_gpu`), max context using GPU+RAM offload
(`ctx_max`), the GPU/CPU layer split at the planning context, and a verdict
(gpu / offload / no / unknown). Everything is a planning estimate; the real fit test is a
launch (Milestone 2).
"""

import json
from pathlib import Path
from typing import Any, cast

from llb.contracts import ModelPlanRow, ModelSpec

MIB = 1024 * 1024
KV_ELEM_BYTES = 2  # fp16 KV cache element
EMBED_BPW = 16.0  # high-precision part (embeddings/norms) stays bf16/fp16 under partial quant

# Bits-per-weight for common quantizations (GGUF k-quants, plain dtypes, served formats).
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
    "q2_k": 3.0,
}

# Defaults (MiB) -- conservative headroom so the plan is honest, not optimistic.
DEFAULT_VRAM_RESERVE = 1024
DEFAULT_RAM_RESERVE = 2048
DEFAULT_OVERHEAD = 512
MIN_USABLE_CTX = 512

VERDICT_GPU = "gpu"  # fits fully in VRAM at the planning context
VERDICT_OFFLOAD = "offload"  # runs only by splitting layers to CPU RAM
VERDICT_NO = "no"  # does not fit even in VRAM + RAM
VERDICT_UNKNOWN = "unknown"  # missing the spec fields needed to plan


def resolve_bpw(spec: ModelSpec) -> float | None:
    if spec.get("bpw") is not None:
        return float(spec["bpw"])
    quant = str(spec.get("quant", "")).lower()
    return QUANT_BPW.get(quant)


def weights_mib(params_b: float, bpw: float) -> float:
    """Flat estimate: every weight at one bpw. Embedding-blind (kept for the no-arch fallback)."""
    return params_b * 1e9 * bpw / 8 / MIB


def embedding_params(vocab_size: int, hidden_size: int, tied: bool = True) -> float:
    """Params in the token embedding table (+ the output head when it is NOT tied)."""
    return vocab_size * hidden_size * (1 if tied else 2)


# Quant formats that keep the embedding / lm_head in high precision and quantize only the
# linear layers (so the embedding premium applies). GGUF k-quants quantize the embedding too,
# so the premium does NOT apply there; bf16/fp16/fp32 are already uniform.
PARTIAL_QUANT_FORMATS = {"w4a16", "int4", "awq", "gptq", "fp8"}


def hi_precision_params(spec: ModelSpec) -> float:
    """Count of weights that stay high-precision under a partial quant, in absolute params.

    An explicit `hi_precision_params_b` always wins (it can capture architecture quirks the
    vocab formula misses, e.g. Gemma 3n Per-Layer Embeddings). Otherwise, ONLY for a partial
    quant that keeps the embedding in fp16, it is the token embedding (plus the untied output
    head) from `vocab_size` x `hidden_size`. 0 when unknown or when the format quantizes
    everything uniformly (GGUF k-quants, bf16, fp32).
    """
    override = spec.get("hi_precision_params_b")
    if override is not None:
        return max(0.0, float(override) * 1e9)
    if str(spec.get("quant", "")).lower() not in PARTIAL_QUANT_FORMATS:
        return 0.0
    vocab = spec.get("vocab_size")
    hidden = spec.get("hidden_size")
    if vocab and hidden:
        tied = bool(spec.get("tie_word_embeddings", True))
        return embedding_params(int(vocab), int(hidden), tied)
    return 0.0


def weights_mib_detailed(
    params_b: float, quant_bpw: float, hi_params: float, embed_bpw: float = EMBED_BPW
) -> float:
    """Embedding-aware weights (MiB): high-precision mass at `embed_bpw`, the rest at `quant_bpw`.

    `embed_bpw` is floored at the quant bpw, so a full-precision checkpoint (bf16/fp32) reduces
    to the flat estimate (the embedding is not magically cheaper than the body).
    """
    total = params_b * 1e9
    hi = min(max(0.0, hi_params), total)
    hi_bpw = max(embed_bpw, quant_bpw)
    body_bytes = (total - hi) * quant_bpw / 8
    hi_bytes = hi * hi_bpw / 8
    return (body_bytes + hi_bytes) / MIB


# --- arch enrichment from a cached config.json (best-effort; never downloads) --------------


def arch_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Pull planning arch fields from a HF `config.json` dict (handles nested `text_config`)."""
    text = config["text_config"] if isinstance(config.get("text_config"), dict) else {}
    out: dict[str, Any] = {}
    for key, dest in (
        ("vocab_size", "vocab_size"),
        ("hidden_size", "hidden_size"),
        ("num_hidden_layers", "n_layers"),
        ("sliding_window", "sliding_window"),
        ("sliding_window_pattern", "sliding_window_pattern"),
    ):
        value = text.get(key, config.get(key))
        if isinstance(value, int) and not isinstance(value, bool):
            out[dest] = value
    tie = config.get("tie_word_embeddings", text.get("tie_word_embeddings"))
    if isinstance(tie, bool):
        out["tie_word_embeddings"] = tie
    # Newer Gemma configs drop `sliding_window_pattern` for an explicit `layer_types` list
    # (e.g. "sliding_attention" / "full_attention" per layer); derive the period from it.
    layer_types = text.get("layer_types", config.get("layer_types"))
    if "sliding_window_pattern" not in out and isinstance(layer_types, list) and layer_types:
        full = sum(1 for t in layer_types if t == "full_attention")
        if 0 < full < len(layer_types):
            out["sliding_window_pattern"] = max(2, len(layer_types) // full)
    return out


def cached_config_path(repo_id: str) -> Path | None:
    """Path to a model's `config.json` already in the local HF cache, or None. Never downloads."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return None
    try:
        hit = try_to_load_from_cache(repo_id, "config.json")
    except Exception:
        return None
    return Path(hit) if isinstance(hit, str) and Path(hit).is_file() else None


def enrich_arch(spec: ModelSpec, *, override: bool = False) -> ModelSpec:
    """Merge arch fields (vocab/hidden/n_layers/tie/sliding-window) from a cached `config.json`.

    Best-effort and offline, only for HF repo ids ("org/name") whose weights are already cached
    (Ollama / hf.co tags are skipped). By default the config only FILLS GAPS -- curated spec
    values win. With `override=True` the cached config (the real served architecture) OVERRIDES
    the curated values, so a hand-curated guess can never silently misprice a model whose true
    config is on disk.
    """
    wanted = ("vocab_size", "hidden_size", "n_layers", "tie_word_embeddings")
    if not override and all(spec.get(k) is not None for k in wanted):
        return spec
    source = spec.get("source", "")
    if not source or source.count("/") != 1 or source.startswith("hf.co/"):
        return spec
    path = cached_config_path(source)
    if path is None:
        return spec
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return spec
    if not isinstance(config, dict):
        return spec
    merged: dict[str, Any] = dict(spec)
    for key, value in arch_from_config(config).items():
        if override or merged.get(key) is None:
            merged[key] = value
    return cast(ModelSpec, merged)


def kv_mib_per_token(n_layers: int, kv_dim: int) -> float:
    """MiB of KV cache per token (K and V, all layers, batch 1)."""
    return 2 * n_layers * kv_dim * KV_ELEM_BYTES / MIB


def attention_layer_split(n_layers: int, sliding_window_pattern: int) -> tuple[int, int]:
    """(full_layers, sliding_layers) for a Gemma-style interleaved schedule where every
    `pattern`-th layer is full-attention. `pattern` <= 1 -> every layer is full attention."""
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
    """Total KV-cache MiB at `ctx` tokens. With Gemma-style sliding-window attention the sliding
    layers cache at most `sliding_window` tokens (KV stops growing past the window) while the
    periodic full-attention layers cache the whole context. Absent sliding fields -> full
    attention on every layer (the linear `kv_mib_per_token` x ctx)."""
    per_tok_layer = kv_mib_per_token(1, kv_dim)
    if not sliding_window or not sliding_window_pattern or sliding_window_pattern <= 1:
        return per_tok_layer * n_layers * ctx
    full, sliding = attention_layer_split(n_layers, sliding_window_pattern)
    return per_tok_layer * full * ctx + per_tok_layer * sliding * min(ctx, sliding_window)


def max_context(
    budget_mib: float, w_mib: float, overhead_mib: float, per_tok_mib: float, cap: int
) -> int:
    """Largest context (<= cap) whose footprint fits in `budget_mib`. 0 if weights don't fit."""
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
    """Largest context (<= cap) whose weights + KV fit `budget_mib`, sliding-window-aware.

    For full attention this reduces to `max_context` with `per_tok = kv_mib_per_token`. With
    sliding-window layers the KV is PIECEWISE in ctx: below the window every layer grows; above
    it only the full-attention layers grow (the sliding layers are pinned at the window), so a
    long context costs far less KV -- which is exactly why a 12B Gemma fits a 16 GB card."""
    avail = budget_mib - w_mib - overhead_mib
    if avail <= 0:
        return 0
    per_tok_layer = kv_mib_per_token(1, kv_dim)
    if per_tok_layer <= 0:
        return 0
    if not sliding_window or not sliding_window_pattern or sliding_window_pattern <= 1:
        return max(0, min(cap, int(avail / (per_tok_layer * n_layers))))
    full, sliding = attention_layer_split(n_layers, sliding_window_pattern)
    # Region 1: ctx <= window, all layers grow linearly.
    linear_ctx = int(avail / (per_tok_layer * n_layers))
    if linear_ctx <= sliding_window:
        return max(0, min(cap, linear_ctx))
    # Region 2: ctx > window -- the sliding layers are pinned at `window`, only full layers grow.
    after_sliding = avail - per_tok_layer * sliding * sliding_window
    if full == 0 or after_sliding <= 0:
        return max(0, min(cap, sliding_window))
    return max(0, min(cap, int(after_sliding / (per_tok_layer * full))))


def gpu_layers(
    vram_usable_mib: float, overhead_mib: float, w_mib: float, kv_at_ctx_mib: float, n_layers: int
) -> int:
    """How many of `n_layers` (weights + their KV) fit in VRAM; the rest go to CPU RAM."""
    per_layer = (w_mib + kv_at_ctx_mib) / n_layers
    if per_layer <= 0:
        return n_layers
    fit = int((vram_usable_mib - overhead_mib) / per_layer)
    return max(0, min(n_layers, fit))


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
    """Plan one model on the host. Returns a row dict (see module docstring)."""
    row: ModelPlanRow = {
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

    bpw = resolve_bpw(spec)
    params_b = spec.get("params_b")
    if bpw is None or params_b is None:
        row["note"] = "add params_b + quant/bpw to plan"
        return row
    embed_bpw = float(spec.get("embed_bpw") or EMBED_BPW)
    w = weights_mib_detailed(float(params_b), bpw, hi_precision_params(spec), embed_bpw)
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

    sliding_window = spec.get("sliding_window")
    sliding_pattern = spec.get("sliding_window_pattern")
    row["ctx_gpu"] = max_context_for_kv(
        vram_usable,
        w,
        overhead_mib,
        n_layers,
        kv_dim,
        cap,
        sliding_window=sliding_window,
        sliding_window_pattern=sliding_pattern,
    )
    row["ctx_max"] = max_context_for_kv(
        total,
        w,
        overhead_mib,
        n_layers,
        kv_dim,
        cap,
        sliding_window=sliding_window,
        sliding_window_pattern=sliding_pattern,
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
    kv_at = kv_mib_at_context(
        n_layers,
        kv_dim,
        planning_ctx,
        sliding_window=sliding_window,
        sliding_window_pattern=sliding_pattern,
    )
    gl = gpu_layers(vram_usable, overhead_mib, w, kv_at, n_layers)
    row["gpu_layers"] = gl
    row["verdict"] = VERDICT_GPU if gl >= n_layers else VERDICT_OFFLOAD
    row["note"] = f"plan @ ctx={planning_ctx}"
    return row


def plan_models(
    models: list[ModelSpec], vram_mib: int, ram_mib: int, **kwargs: Any
) -> list[ModelPlanRow]:
    return [plan_model(m, vram_mib, ram_mib, **kwargs) for m in models]


def _gb(mib: float | None) -> str:
    return "-" if mib is None else f"{mib / 1024:.1f}"


def format_plan(rows: list[ModelPlanRow], vram_mib: int, ram_mib: int) -> str:
    """ASCII table of the plan."""
    headers = [
        "model",
        "backend",
        "params",
        "quant",
        "wt_GB",
        "ctx_gpu",
        "ctx_max",
        "gpu/total",
        "verdict",
    ]

    def fmt(row: ModelPlanRow) -> list[str]:
        n_layers = row["n_layers"]
        split = "-" if not n_layers else f"{row['gpu_layers']}/{n_layers}"
        return [
            row["name"],
            row["backend"],
            "-" if row["params_b"] is None else f"{row['params_b']}B",
            row["quant"] or "-",
            _gb(row["weights_mib"]),
            str(row["ctx_gpu"]) if row["ctx_gpu"] else "-",
            str(row["ctx_max"]) if row["ctx_max"] else "-",
            split,
            row["verdict"],
        ]

    table = [fmt(r) for r in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in table)) if table else len(h) for i, h in enumerate(headers)
    ]
    out = [
        f"host budget: VRAM {vram_mib} MiB + RAM {ram_mib} MiB "
        f"(usable after reserves; weights + KV must fit the combined budget)",
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for r in table:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)
