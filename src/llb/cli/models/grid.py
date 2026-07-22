"""Grid-expansion + backend-readiness helpers for the sweep command."""

from typing import Any

import typer

from llb.backends.readiness import local_backend_ready
from llb.core.config import RunConfig
from llb.core.contracts.models import ResolvedModel

_local_backend_ready = local_backend_ready


def _sweep_cell_overrides(
    resolution: ResolvedModel, telemetry: bool, max_model_len: int
) -> dict[str, Any] | None:
    """Build RunConfig overrides for one resolved model, or None when not runnable."""
    if not resolution["chosen_backend"]:
        return None
    overrides: dict[str, Any] = {
        "model": resolution["chosen_source"],
        "backend": resolution["chosen_backend"],
        "measure_telemetry": telemetry,
        "run_name": f"sweep-{resolution['name']}",
    }
    if resolution["chosen_backend"] == "vllm":
        overrides["max_model_len"] = max_model_len
    elif resolution["chosen_backend"] == "llamacpp":
        from llb.backends.resolver import llamacpp_offload_split

        ngl = llamacpp_offload_split(resolution)
        if ngl is not None:
            overrides["n_gpu_layers"] = ngl
            typer.echo(f"[sweep] {resolution['name']}: llama.cpp offload split -ngl={ngl}")
    return overrides


# Query-time --rag-grid axes: value parser + validity predicate per supported key. Only
# query-time knobs belong here (they retrieve against the SAME index, so no re-index per cell).
# `rerank_candidates` (rerank-context-order): 0 == reranker off; a positive depth enables the
# sweep-level `--reranker` cross-encoder with that candidate pool.
_RAG_GRID_AXES: dict[str, tuple[Any, Any]] = {
    "top_k": (int, lambda v: v >= 1),
    "fusion_weight": (float, lambda v: 0.0 <= v <= 1.0),
    "graph_weight": (float, lambda v: 0.0 <= v <= 1.0),
    "graph_fusion_candidates": (int, lambda v: v >= 1),
    "rerank_candidates": (int, lambda v: v >= 0),
}
_RAG_GRID_USAGE = (
    "--rag-grid must look like 'top_k=3,5,8', 'top_k=3,5;fusion_weight=0.4,0.6', "
    "'graph_weight=0,0.3', or 'rerank_candidates=0,30' (0 == reranker off)"
)


def _parse_grid_axis(part: str, seen: set[str]) -> tuple[str, list[Any]]:
    """Parse one `key=v1,v2,...` grid axis, validating the key, types, and value ranges."""
    key, sep, raw = part.partition("=")
    key = key.strip()
    if key not in _RAG_GRID_AXES or not sep or not raw.strip():
        raise typer.BadParameter(_RAG_GRID_USAGE)
    if key in seen:
        raise typer.BadParameter(f"--rag-grid axis '{key}' given twice")
    cast, valid = _RAG_GRID_AXES[key]
    try:
        values = [cast(v) for v in raw.split(",") if v.strip()]
    except ValueError as exc:
        raise typer.BadParameter(
            f"--rag-grid {key} values must be {cast.__name__}s: {raw!r}"
        ) from exc
    values = list(dict.fromkeys(values))  # de-dupe, preserve order
    if not values or not all(valid(v) for v in values):
        raise typer.BadParameter(f"--rag-grid {key} values out of range: {raw!r}")
    return key, values


def _parse_rag_grid(spec: str | None) -> list[dict[str, Any]]:
    """Parse an opt-in RAG-config grid into per-cell override dicts (axes cross-multiplied).

    Returns `[{}]` (keep the manifest's single config) when no grid is given, so the default
    sweep is unchanged. Supported axes (`;`-separated): `top_k` (retrieval depth) and
    `fusion_weight` (hybrid dense/lexical RRF share; the index must be built with
    `build-index --retrieval-mode hybrid`). Index-time knobs (chunk_size/overlap) are out of
    scope because they need rebuilt indexes.
    """
    if not spec:
        return [{}]
    axes: list[tuple[str, list[Any]]] = []
    for part in spec.split(";"):
        axes.append(_parse_grid_axis(part, {seen for seen, _ in axes}))
    points = [{}]  # type: list[dict[str, Any]]
    for key, values in axes:
        points = [{**point, key: value} for point in points for value in values]
    return points


_GRID_SUFFIX_PREFIX = {
    "top_k": "k",
    "fusion_weight": "w",
    "graph_weight": "gw",
    "graph_fusion_candidates": "gc",
    "rerank_candidates": "r",
}


def _grid_cells(
    base: RunConfig,
    overrides: dict[str, Any],
    rag_grid: list[dict[str, Any]],
    reranker: str | None = None,
) -> list[RunConfig]:
    """One revalidated RunConfig per grid point for a resolved model (a single cell when no grid).

    Every grid knob is a `RunConfig` field and therefore part of the cell fingerprint, so
    distinct grid points get distinct resume keys; the `-k<top_k>`/`-w<fusion_weight>`/
    `-r<rerank_candidates>` run-name suffix only makes the sweep log readable. A `fusion_weight`
    point implies `retrieval_mode=hybrid` (the knob is dead outside hybrid fusion). A
    `rerank_candidates` point of 0 turns the reranker OFF; a positive depth turns it on with
    the sweep-level `reranker` cross-encoder id.
    """
    cells: list[RunConfig] = []
    for point in rag_grid:
        cell = dict(overrides)
        _apply_grid_point(cell, point, reranker)
        suffix = _grid_point_suffix(point)
        if suffix:
            cell["run_name"] = f"{overrides['run_name']}{suffix}"
        cells.append(base.with_overrides(**cell))
    return cells


def _grid_point_suffix(point: dict[str, Any]) -> str:
    """Readable `-k8-w0.5` style run-name suffix for one grid point."""
    return "".join(
        f"-{_GRID_SUFFIX_PREFIX[key]}{value:g}"
        if isinstance(value, float)
        else f"-{_GRID_SUFFIX_PREFIX[key]}{value}"
        for key, value in point.items()
    )


def _apply_grid_point(cell: dict[str, Any], point: dict[str, Any], reranker: str | None) -> None:
    """Translate one grid point into RunConfig overrides (rerank/hybrid implications included)."""
    for key, value in point.items():
        if key == "rerank_candidates":
            if value == 0:
                cell["reranker"] = None
            else:
                cell["reranker"] = reranker
                cell["rerank_candidates"] = value
            continue
        cell[key] = value
    if "fusion_weight" in point:
        cell["retrieval_mode"] = "hybrid"
    if "graph_weight" in point or "graph_fusion_candidates" in point:
        cell["retrieval_backend"] = "fused"
