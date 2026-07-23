"""Retrieval-quality comparison command (compare-retrieval across stores)."""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.cli.rag.compare_stores import _compare_vector_corpus_root


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
        help="compare dense vs hybrid (BM25+RRF) vs hybrid+lemmas plus the oracle-doc-filter "
        "headroom row over one embedded corpus (the sibling corpus/ of --goldset when present); "
        "the hybrid store persists under $DATA_DIR/llb/rag/hybrid/",
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
    out: Optional[Path] = typer.Option(None, help="write the JSON comparison report here"),
) -> None:
    """Compare retrieval backends -- or chunking strategies, or hybrid fusion -- on one gold set.

    Default: scores each BUILT backend (FAISS vs graph/local_khop vs graph/global_community) on
    the SAME items (a backend whose store is not built is skipped). With `--strategies` it instead
    builds one store per CHUNKING strategy (same corpus + pinned embedder) and ranks the chunkers,
    so the best chunker is demonstrated per corpus. With `--hybrid` it demonstrates (not assumes)
    per corpus whether dense+BM25 RRF fusion beats dense-only, what Ukrainian lemmatization adds,
    and how much recall headroom perfect document routing would buy. `--reranker` adds a reranked
    twin row per compared row (rerank-context-order). Answer-quality comparison rides
    `run-eval --retrieval-backend ...` (it needs a model).
    """
    import json

    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag.compare import (
        add_rerank_rows,
        compare_retrieval,
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
    report = compare_retrieval(
        stores,
        compare_items,
        k,
        slice_labels=aligned_question_types(cfg.goldset_path, [it.id for it in items]),
    )
    census = duplicate_census(stores)
    if census:
        report["duplicates"] = census
    if noise_floor:
        from llb.rag.noise_floor import DEFAULT_REPLICATES, measure_noise_floor

        report["noise_floor"] = measure_noise_floor(
            stores, compare_items, k, replicates=noise_floor_replicates or DEFAULT_REPLICATES
        )
    typer.echo(format_comparison(report))
    _echo_stage_latencies(stores)
    if out is not None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[compare-retrieval] wrote report -> {out}")


def _build_compare_stores(
    cfg: Any, strategies: Optional[str], hybrid: bool, compare_items: list[Any]
) -> dict[str, Any]:
    """The label -> store map to compare: per-strategy builds, hybrid rows, or built backends."""
    from llb.rag.comparison_builders import (
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
