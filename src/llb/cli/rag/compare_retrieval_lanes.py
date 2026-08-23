"""Lane assembly for `compare-retrieval`: which rows are built, twinned, paired, and reported.

The command module owns the option surface; this one owns what those options MEAN as rows -- the
mode a comparison is in, the stores each mode builds, the twin rows layered over them, the incumbent
each mode is paired against, which rows a verdict may name, and the sidecars the finished lanes
report beside their scores.
"""

from typing import Any, Optional

import typer

from llb.cli.rag.compare_stores import resolve_paired_baseline

# The comparison modes, in the order the error message names them. Each builds its own row set over
# one corpus, so two at once would score two different row sets into one report.
MODE_FLAGS = ("--strategies", "--sizes", "--hybrid")


def refuse_two_modes(strategies: Optional[str], sizes: Optional[str], hybrid: bool) -> None:
    """Exit 2 when more than one comparison mode is selected."""
    selected = [name for name, on in zip(MODE_FLAGS, (strategies, sizes, hybrid)) if on]
    if len(selected) > 1:
        typer.echo(f"[error] {', '.join(selected)} are mutually exclusive", err=True)
        raise typer.Exit(code=2)


def build_compare_stores(
    cfg: Any,
    strategies: Optional[str],
    sizes: Optional[str],
    hybrid: bool,
    compare_items: list[Any],
) -> dict[str, Any]:
    """The label -> store map: per-size or per-strategy builds, hybrid rows, or built backends."""
    from llb.rag.comparison.builders import (
        build_chunking_comparison,
        build_hybrid_comparison,
        load_compare_stores,
    )

    if sizes:
        stores = _size_stores(cfg, sizes)
    elif strategies:
        selected = [s.strip() for s in strategies.split(",") if s.strip()]
        stores = _built(
            lambda: build_chunking_comparison(cfg, selected, stores_root=cfg.index_dir())
        )
        typer.echo(f"[compare-retrieval] per-strategy stores saved under {cfg.index_dir()}/")
    elif hybrid:
        stores = build_hybrid_comparison(cfg, compare_items, stores_root=cfg.index_dir())
        typer.echo(f"[compare-retrieval] hybrid store saved under {cfg.index_dir()}/hybrid/")
    else:
        stores = load_compare_stores(cfg)
    if not stores:
        typer.echo(
            "[error] no retrieval backend is built (run build-index / build-graph)", err=True
        )
        raise typer.Exit(code=2)
    return stores


def _size_stores(cfg: Any, sizes: str) -> dict[str, Any]:
    """One store per requested chunk `size` under the config's own strategy."""
    from llb.rag.comparison.builders import build_chunk_size_comparison

    try:
        selected = [int(value.strip()) for value in sizes.split(",") if value.strip()]
    except ValueError:
        typer.echo(f"[error] --sizes takes integers, got {sizes!r}", err=True)
        raise typer.Exit(code=2) from None
    stores = _built(lambda: build_chunk_size_comparison(cfg, selected, stores_root=cfg.index_dir()))
    typer.echo(f"[compare-retrieval] per-size stores saved under {cfg.index_dir()}/")
    return stores


def _built(build: Any) -> dict[str, Any]:
    """Run a builder, turning its rejection of an unknown lane into a usage error."""
    try:
        return dict(build())
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None


def add_twin_rows(
    stores: dict[str, Any], reranker: Optional[str], rerank_candidates: Optional[int], stitch: bool
) -> dict[str, Any]:
    """Layer the optional twin rows over the built lanes: reranked first, then stitched.

    Order matters and is the operator's reading of it: stitching runs on what a lane FINALLY
    delivers, so a reranked lane's stitched twin reflows the reranked top-k, not the pre-rerank pool.
    """
    from llb.rag.comparison.rows import add_rerank_rows, add_stitch_rows

    if reranker:
        from llb.rag.rerank import DEFAULT_RERANK_CANDIDATES, CrossEncoderReranker

        stores = add_rerank_rows(
            stores, CrossEncoderReranker(reranker), rerank_candidates or DEFAULT_RERANK_CANDIDATES
        )
    return add_stitch_rows(stores) if stitch else stores


def comparison_baseline(
    stores: dict[str, Any],
    requested: str | None,
    cfg: Any,
    strategies: str | None,
    sizes: str | None,
    hybrid: bool,
) -> str:
    """Resolve a stable, mode-aware baseline before any item is retrieved.

    Each mode has its own incumbent -- the shipped retrieval path of that comparison -- and the
    resolution/validation itself is shared with `compare-vector-stores`. In `--sizes` mode the
    incumbent is the config's OWN size, so a swept size is read against the size in production
    rather than against whichever value the operator happened to list first.
    """
    from llb.rag.comparison.models import size_row_label

    if sizes:
        preferred = (size_row_label(cfg.strategy, cfg.chunk_size),)
    elif hybrid:
        preferred = ("dense",)
    elif strategies:
        preferred = ("recursive",)
    else:
        preferred = ("faiss",)
    return resolve_paired_baseline(stores, requested, preferred)


def verdict_lanes(stores: dict[str, Any], hybrid: bool) -> list[str]:
    """Return rows a verdict can name: oracle/lexical diagnostics and stitched twins cannot.

    A stitched twin retrieves exactly what its base lane retrieves, so it ties the baseline on
    every metric the verdict is decided on and would win the tie-break on a compressed `mrr`. It
    is a REPORTED lever, read on `intact@k` against `chars@k`, never an adoption candidate.
    """
    from llb.rag.comparison.models import (
        RERANK_ROW_SUFFIX,
        ROW_LEXICAL,
        ROW_ORACLE_DOC,
        STITCH_ROW_SUFFIX,
    )

    excluded = {ROW_ORACLE_DOC}
    if hybrid:
        excluded.update({ROW_LEXICAL, f"{ROW_LEXICAL}{RERANK_ROW_SUFFIX}"})
    return [
        lane for lane in stores if lane not in excluded and not lane.endswith(STITCH_ROW_SUFFIX)
    ]


def attach_diagnostics(
    report: Any,
    stores: dict[str, Any],
    compare_items: list[Any],
    k: int,
    *,
    noise_floor: bool,
    noise_floor_replicates: Optional[int],
) -> None:
    """Attach the sidecars the scored lanes produced: stitching, duplicates, measurement floor.

    Each is omitted when its lanes produced nothing, so an absent key means "not measured" rather
    than "measured as zero".
    """
    from llb.rag.comparison.rows import duplicate_census, stitch_report

    stitched = stitch_report(report, stores)
    if stitched:
        report["stitching"] = stitched
    census, census_kept = duplicate_census(stores)
    if census:
        report["duplicates"] = census
        if census_kept:
            report["duplicates_kept"] = census_kept
    if noise_floor:
        from llb.rag.noise_floor.measure import DEFAULT_REPLICATES, measure_noise_floor

        report["noise_floor"] = measure_noise_floor(
            stores, compare_items, k, replicates=noise_floor_replicates or DEFAULT_REPLICATES
        )


def echo_stage_latencies(stores: dict[str, Any]) -> None:
    """Print per-store retrieve/rerank stage latency when the store measured it."""
    for label, store in sorted(stores.items()):
        latency = getattr(store, "mean_stage_latency", None)
        if callable(latency):
            stages = latency()
            typer.echo(
                f"[compare-retrieval] {label}: mean/query retrieve "
                f"{stages['retrieve_s'] * 1000:.1f} ms + rerank {stages['rerank_s'] * 1000:.1f} ms"
            )
