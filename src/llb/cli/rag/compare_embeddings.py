"""Embedder bake-off command (`compare-embeddings`): rank encoders on one gold set."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.cli.rag.compare_stores import _compare_vector_corpus_root, _dir_size_bytes

if TYPE_CHECKING:
    from llb.core.config import RunConfig
    from llb.prep.frontier_telemetry import ProvenanceLog
    from llb.rag.embedding_bakeoff import StoreBuilder


def _local_store_builder(cfg: "RunConfig", stores_dir: Path) -> "StoreBuilder":
    """Build+save one FAISS store per embedding model, timing the embed pass and measuring size."""
    import time

    from llb.rag.embedding_bakeoff import BuiltStore, slugify_model
    from llb.rag.store import RagStore

    def build(model: str) -> "BuiltStore":
        started = time.perf_counter()
        store = RagStore.build(
            cfg.corpus_root,
            cfg.strategy,
            cfg.chunk_size,
            cfg.chunk_overlap,
            model,
            mode=cfg.retrieval_mode,
            child_size=cfg.child_chunk_size,
            lexical_lemmas=cfg.lexical_lemmas,
        )
        embed_seconds = time.perf_counter() - started
        out_dir = stores_dir / slugify_model(model)
        store.save(out_dir)
        device = None
        resolve = getattr(store.embedder, "_resolve_device", None)
        if callable(resolve):
            device = resolve()
        return BuiltStore(
            store=store,
            embed_seconds=embed_seconds,
            index_bytes=_dir_size_bytes(out_dir),
            device=device,
        )

    return build


def _api_store_builder(
    cfg: "RunConfig", stores_dir: Path, log: "ProvenanceLog", max_usd: Optional[float]
) -> "StoreBuilder":
    """Build+save the API-embedded store (corpus egress); records cost from the litellm embed log."""
    import time

    from llb.rag.api_embedder import ApiEmbedder, litellm_embed
    from llb.rag.embedding_bakeoff import KIND_API, BuiltStore, slugify_model
    from llb.rag.store import RagStore

    def build(model: str) -> "BuiltStore":
        embedder = ApiEmbedder(model, litellm_embed(model, log=log, max_usd=max_usd))
        started = time.perf_counter()
        store = RagStore.build(
            cfg.corpus_root,
            cfg.strategy,
            cfg.chunk_size,
            cfg.chunk_overlap,
            model,
            mode=cfg.retrieval_mode,
            child_size=cfg.child_chunk_size,
            embedder=embedder,
            lexical_lemmas=cfg.lexical_lemmas,
        )
        embed_seconds = time.perf_counter() - started
        out_dir = stores_dir / slugify_model(model)
        store.save(out_dir)
        return BuiltStore(
            store=store,
            embed_seconds=embed_seconds,
            index_bytes=_dir_size_bytes(out_dir),
            kind=KIND_API,
            cost_usd=log.summary()["total_cost_usd"],
        )

    return build


@app.command("compare-embeddings")
def compare_embeddings_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    corpus_root: Optional[Path] = typer.Option(
        None,
        help="corpus to build for each candidate; defaults to the sibling corpus/ of --goldset",
    ),
    models: str = typer.Option(
        "",
        help="comma-separated local embedding model ids; empty uses the default UA candidate set",
    ),
    k: int = typer.Option(10, help="recall@k / MRR cutoff"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    api_model: Optional[str] = typer.Option(
        None,
        help="opt-in API embedder row (e.g. cohere/embed-multilingual-v3.0); full corpus EGRESS -- "
        "needs --data-classification open + interactive consent, honors --max-usd",
    ),
    data_classification: Optional[str] = typer.Option(
        None, help="corpus data classification; must be 'open' to enable --api-model"
    ),
    max_usd: Optional[float] = typer.Option(
        None, help="hard budget cap for the --api-model egress lane (USD)"
    ),
    yes: bool = typer.Option(
        False, "--yes", help="skip the interactive egress consent prompt for --api-model"
    ),
    noise_floor: bool = typer.Option(
        False,
        "--noise-floor",
        help="also measure the MEASUREMENT FLOOR per candidate (see compare-retrieval) and state "
        "whether the recommended embedder's lead over the runner-up clears it",
    ),
    noise_floor_replicates: Optional[int] = typer.Option(
        None, help="--noise-floor: jitter replicates per candidate (default 64)"
    ),
    out: Optional[Path] = typer.Option(
        None, help="write report.md here (default: $DATA_DIR/compare-embeddings/<ts>/report.md)"
    ),
) -> None:
    """Rank candidate embedders for Ukrainian RAG on one gold set (recall@k / MRR + throughput).

    Builds one store per candidate over the SAME corpus + chunking, scores the source-span metric,
    and writes a ranked report.md with the recommended embedder. Heavy store builds stay outside
    quick CI. The opt-in --api-model row is corpus egress (bake-off evidence only; scored retrieval
    stays local) and is refused unless --data-classification open plus explicit consent.
    """
    from llb.bench.common import new_run_timestamp
    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.prep.frontier_telemetry import ProvenanceLog
    from llb.rag.embedding_bakeoff import (
        DEFAULT_LOCAL_CANDIDATES,
        run_bakeoff,
    )
    from llb.rag.embedding_bakeoff_report import format_report, render_markdown

    cfg = load_config(
        config,
        goldset_path=goldset,
        corpus_root=_compare_vector_corpus_root(goldset, corpus_root),
    )
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    bakeoff_items = [(it.question, spans_as_dicts(it)) for it in items]
    local_models = [m.strip() for m in models.split(",") if m.strip()] or DEFAULT_LOCAL_CANDIDATES

    _, run_ts = new_run_timestamp()
    run_dir = cfg.data_dir / "compare-embeddings" / run_ts
    stores_dir = run_dir / "stores"
    stores_dir.mkdir(parents=True, exist_ok=True)

    egress_log = ProvenanceLog()

    def consent() -> bool:
        if yes:
            return True
        return typer.confirm(
            f"[compare-embeddings] embed the corpus at {cfg.corpus_root} through {api_model} "
            "(full corpus egress to a hosted API). Proceed?"
        )

    report = run_bakeoff(
        bakeoff_items,
        k,
        corpus_root=str(cfg.corpus_root),
        local_models=local_models,
        build_local=_local_store_builder(cfg, stores_dir),
        api_model=api_model,
        build_api=_api_store_builder(cfg, stores_dir, egress_log, max_usd),
        data_classification=data_classification,
        consent=consent,
        noise_floor=noise_floor,
        noise_floor_replicates=noise_floor_replicates,
    )

    typer.echo(format_report(report))
    report_path = out if out is not None else run_dir / "report.md"
    report_path.write_text(render_markdown(report), encoding="utf-8")
    typer.echo(f"[compare-embeddings] wrote report -> {report_path}")
