"""Retrieval-quality comparison command (compare-retrieval across stores)."""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.cli.rag.compare_stores import _compare_vector_corpus_root, resolve_paired_baseline
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
)


@app.command("compare-retrieval")
def compare_retrieval_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, help="recall@k / MRR cutoff"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    strategies: Optional[str] = typer.Option(
        None,
        "--strategies",
        help="comma-separated CHUNKING strategies to compare instead of the built backends "
        "(builds one FAISS store per strategy over the corpus -- the sibling corpus/ of "
        "--goldset when present -- and persists each under $DATA_DIR/llb/rag/<strategy>/)",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid",
        help="compare dense vs lexical (BM25 alone) vs hybrid (BM25+RRF) vs hybrid+lemmas plus "
        "the oracle-doc-filter headroom row over one embedded corpus (the sibling corpus/ of "
        "--goldset when present); the hybrid store persists under $DATA_DIR/llb/rag/hybrid/",
    ),
    fusion_weight: Optional[float] = typer.Option(
        None, help="hybrid rows: dense share of the weighted RRF (0..1; default 0.5)"
    ),
    graph_weight: Optional[float] = typer.Option(
        None, help="fused rows: graph share of weighted RRF (0..1; default 0.3)"
    ),
    reranker: Optional[str] = typer.Option(
        None,
        help="add a '<row>+rerank' twin per compared row: retrieve --rerank-candidates, "
        "rerank with this local cross-encoder (HF id), keep k -- the pre/post-rerank "
        "recall@k/MRR delta plus the measured rerank latency",
    ),
    rerank_candidates: Optional[int] = typer.Option(
        None, help="rerank rows: candidate pool depth fed into the reranker (default 30)"
    ),
    duplicate_tier: Optional[str] = typer.Option(
        None,
        "--duplicate-tier",
        help="duplicate-collapse tier for the stores this run BUILDS (--strategies / --hybrid): "
        "exact (default) | normalized | masked -- see `measure-duplicate-residue` for the "
        "residue each tier would take",
    ),
    noise_floor: bool = typer.Option(
        False,
        "--noise-floor",
        help="also measure the MEASUREMENT FLOOR: re-rank each lane's candidates under "
        "numeric score noise of the measured between-process amplitude and report the "
        "resulting recall@k / MRR band, so a delta smaller than the floor reads as noise",
    ),
    noise_floor_replicates: Optional[int] = typer.Option(
        None, help="--noise-floor: jitter replicates per lane (default 64)"
    ),
    baseline: Optional[str] = typer.Option(
        None,
        help="paired baseline lane (defaults by mode: recursive, dense, or faiss)",
    ),
    resamples: int = typer.Option(
        DEFAULT_RESAMPLES, min=0, help="paired percentile-bootstrap resamples"
    ),
    confidence: float = typer.Option(
        DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="paired bootstrap confidence level"
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="paired bootstrap seed"),
    out: Optional[Path] = typer.Option(None, help="write the JSON comparison report here"),
) -> None:
    """Compare retrieval backends -- or chunking strategies, or hybrid fusion -- on one gold set.

    Default: scores each BUILT backend (FAISS vs graph/local_khop vs graph/global_community) on
    the SAME items (a backend whose store is not built is skipped). With `--strategies` it instead
    builds one store per CHUNKING strategy (same corpus + pinned embedder) and ranks the chunkers,
    so the best chunker is demonstrated per corpus. With `--hybrid` it demonstrates (not assumes)
    per corpus whether dense+BM25 RRF fusion beats dense-only, how each lane retrieves ALONE,
    what Ukrainian lemmatization adds, and how much recall headroom perfect document routing
    would buy. `--reranker` adds a reranked
    twin row per compared row (rerank-context-order). Answer-quality comparison rides
    `run-eval --retrieval-backend ...` (it needs a model).
    """
    import json

    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag.comparison.run import compare_retrieval
    from llb.rag.comparison.rows import (
        add_rerank_rows,
        duplicate_census,
        format_comparison,
    )
    from llb.rag.question_types import aligned_question_types

    if strategies and hybrid:
        typer.echo("[error] --strategies and --hybrid are mutually exclusive", err=True)
        raise typer.Exit(code=2)
    cfg = load_config(
        config,
        goldset_path=goldset,
        corpus_root=_compare_vector_corpus_root(goldset, None) if (strategies or hybrid) else None,
        fusion_weight=fusion_weight,
        graph_weight=graph_weight,
        duplicate_tier=duplicate_tier,
    )
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    compare_items = [(it.question, spans_as_dicts(it)) for it in items]
    stores = _build_compare_stores(cfg, strategies, hybrid, compare_items)
    if reranker:
        from llb.rag.rerank import DEFAULT_RERANK_CANDIDATES, CrossEncoderReranker

        stores = add_rerank_rows(
            stores,
            CrossEncoderReranker(reranker),
            rerank_candidates or DEFAULT_RERANK_CANDIDATES,
        )
    try:
        paired_baseline = _comparison_baseline(stores, baseline, strategies, hybrid)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    report = compare_retrieval(
        stores,
        compare_items,
        k,
        slice_labels=aligned_question_types(cfg.goldset_path, [it.id for it in items]),
        item_ids=[it.id for it in items],
        baseline=paired_baseline,
        eligible_lanes=_verdict_lanes(stores, hybrid),
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
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
    typer.echo(format_comparison(report))
    _echo_stage_latencies(stores)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[compare-retrieval] wrote report -> {out}")


def _build_compare_stores(
    cfg: Any, strategies: Optional[str], hybrid: bool, compare_items: list[Any]
) -> dict[str, Any]:
    """The label -> store map to compare: per-strategy builds, hybrid rows, or built backends."""
    from llb.rag.comparison.builders import (
        build_chunking_comparison,
        build_hybrid_comparison,
        load_compare_stores,
    )

    if strategies:
        selected = [s.strip() for s in strategies.split(",") if s.strip()]
        try:
            stores = build_chunking_comparison(cfg, selected, stores_root=cfg.index_dir())
        except ValueError as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(code=2) from None
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


def _echo_stage_latencies(stores: dict[str, Any]) -> None:
    """Print per-store retrieve/rerank stage latency when the store measured it."""
    for label, store in sorted(stores.items()):
        latency = getattr(store, "mean_stage_latency", None)
        if callable(latency):
            stages = latency()
            typer.echo(
                f"[compare-retrieval] {label}: mean/query retrieve "
                f"{stages['retrieve_s'] * 1000:.1f} ms + rerank {stages['rerank_s'] * 1000:.1f} ms"
            )


def _comparison_baseline(
    stores: dict[str, Any],
    requested: str | None,
    strategies: str | None,
    hybrid: bool,
) -> str:
    """Resolve a stable, mode-aware baseline before any item is retrieved.

    Each mode has its own incumbent -- the shipped retrieval path of that comparison -- and the
    resolution/validation itself is shared with `compare-vector-stores`.
    """
    preferred = ("dense",) if hybrid else ("recursive",) if strategies else ("faiss",)
    return resolve_paired_baseline(stores, requested, preferred)


def _verdict_lanes(stores: dict[str, Any], hybrid: bool) -> list[str]:
    """Return deployable rows only: oracle and lexical diagnostics cannot receive ADOPT."""
    from llb.rag.comparison.models import RERANK_ROW_SUFFIX, ROW_LEXICAL, ROW_ORACLE_DOC

    excluded = {ROW_ORACLE_DOC}
    if hybrid:
        excluded.update({ROW_LEXICAL, f"{ROW_LEXICAL}{RERANK_ROW_SUFFIX}"})
    return [lane for lane in stores if lane not in excluded]
