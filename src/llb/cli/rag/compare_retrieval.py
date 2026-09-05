"""Retrieval-quality comparison command (compare-retrieval across stores)."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.cli.rag.compare_stores import _compare_vector_corpus_root
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
    sizes: Optional[str] = typer.Option(
        None,
        "--sizes",
        help="comma-separated chunk SIZES to compare under the config's own strategy (builds one "
        "FAISS store per size over the corpus and persists each under "
        "$DATA_DIR/llb/rag/<strategy>#size<n>/) -- the index-side lever against evidence that "
        "arrives in fragments, priced in the served-context column",
    ),
    stitch: bool = typer.Option(
        False,
        "--stitch",
        help="add a '<row>+stitch' twin per compared row: the SAME top-k with contiguous chunks "
        "of one document merged into one block at assembly time -- the assembly-side lever "
        "against fragmented evidence, which retrieves nothing new and so moves intact@k only",
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
        help="duplicate-collapse tier for the stores this run BUILDS "
        "(--strategies / --sizes / --hybrid): "
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
        help="paired baseline lane (defaults by mode: the config's own <strategy>#size<n>, "
        "recursive, dense, or faiss)",
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
    so the best chunker is demonstrated per corpus. With `--sizes` it holds the strategy and
    varies the chunk `size` cap instead -- the index-side lever for evidence that arrives in
    fragments. With `--hybrid` it demonstrates (not assumes)
    per corpus whether dense+BM25 RRF fusion beats dense-only, how each lane retrieves ALONE,
    what Ukrainian lemmatization adds, and how much recall headroom perfect document routing
    would buy. `--reranker` adds a reranked
    twin row per compared row (rerank-context-order); `--stitch` adds an assembly-time twin that
    merges contiguous retrieved chunks (the lever that reflows evidence without retrieving any).
    Every row is priced in the `chars@k` served-context column. Answer-quality comparison rides
    `run-eval --retrieval-backend ...` (it needs a model).
    """
    from typing import cast

    from llb.artifacts.retrieval_graph.sidecars import write_sidecar
    from llb.core.contracts.common import JsonObject
    from llb.core.contracts.retrieval_graph.comparison import RETRIEVAL_COMPARISON_SCHEMA_ID
    from llb.cli.rag.compare_retrieval_lanes import (
        add_twin_rows,
        attach_diagnostics,
        build_compare_stores,
        comparison_baseline,
        echo_stage_latencies,
        refuse_two_modes,
        verdict_lanes,
    )
    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag.comparison.run import compare_retrieval
    from llb.rag.comparison.rows import format_comparison
    from llb.rag.question_types import aligned_question_types

    refuse_two_modes(strategies, sizes, hybrid)
    cfg = load_config(
        config,
        goldset_path=goldset,
        corpus_root=(
            _compare_vector_corpus_root(goldset, None) if (strategies or sizes or hybrid) else None
        ),
        fusion_weight=fusion_weight,
        graph_weight=graph_weight,
        duplicate_tier=duplicate_tier,
    )
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    compare_items = [(it.question, spans_as_dicts(it)) for it in items]
    stores = build_compare_stores(cfg, strategies, sizes, hybrid, compare_items)
    stores = add_twin_rows(stores, reranker, rerank_candidates, stitch)
    try:
        paired_baseline = comparison_baseline(stores, baseline, cfg, strategies, sizes, hybrid)
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
        eligible_lanes=verdict_lanes(stores, hybrid),
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    attach_diagnostics(
        report,
        stores,
        compare_items,
        k,
        noise_floor=noise_floor,
        noise_floor_replicates=noise_floor_replicates,
    )
    typer.echo(format_comparison(report))
    echo_stage_latencies(stores)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        write_sidecar(out, RETRIEVAL_COMPARISON_SCHEMA_ID, cast(JsonObject, report))
        typer.echo(f"[compare-retrieval] wrote report -> {out}")
