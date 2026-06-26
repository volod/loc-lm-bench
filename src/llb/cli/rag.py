"""RAG index build and retrieval validation commands."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config

if TYPE_CHECKING:
    from llb.config import RunConfig
    from llb.prep.ontology.models import DocExtraction, DocRecord, OntologyCandidate


@app.command("build-index")
def build_index(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    corpus_root: Optional[Path] = typer.Option(None, help="corpus directory to chunk"),
    strategy: Optional[str] = typer.Option(
        None, help="fixed | sentence | recursive | markdown | semantic"
    ),
    size: Optional[int] = typer.Option(None, help="chunk size (chars)"),
    overlap: Optional[int] = typer.Option(None, help="chunk overlap (chars)"),
    embedding_model: Optional[str] = typer.Option(None, help="pinned embedding model"),
    mode: Optional[str] = typer.Option(None, help="flat | parent_child"),
    child_size: Optional[int] = typer.Option(None, help="child chunk size (parent_child mode)"),
) -> None:
    """Chunk + embed the corpus into a FAISS RAG store under the index dir."""
    cfg = load_config(
        config,
        corpus_root=corpus_root,
        strategy=strategy,
        chunk_size=size,
        chunk_overlap=overlap,
        embedding_model=embedding_model,
        retrieval_mode=mode,
        child_chunk_size=child_size,
    )
    from llb.rag.store import RagStore

    store = RagStore.build(
        cfg.corpus_root,
        cfg.strategy,
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.embedding_model,
        mode=cfg.retrieval_mode,
        child_size=cfg.child_chunk_size,
    )
    store.save(cfg.index_dir())
    parents = f", {store.meta['n_parents']} parents" if store.meta["n_parents"] else ""
    typer.echo(
        f"[build-index] {store.meta['n_indexed']} indexed chunks{parents} "
        f"({cfg.strategy}/{cfg.retrieval_mode}, dim {store.meta['dim']}) -> {cfg.index_dir()}"
    )


@app.command("build-graph")
def build_graph_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    bundle: Optional[Path] = typer.Option(
        None,
        help="prepare-goldset draft bundle dir (reads its extraction.jsonl + corpus/ + ontology.json)",
    ),
    extraction: Optional[Path] = typer.Option(
        None, help="explicit M4.4 extraction.jsonl (pair with --corpus-root)"
    ),
    corpus_root: Optional[Path] = typer.Option(
        None, help="corpus dir for --extraction, or to extract fresh when no extraction is given"
    ),
    extract_model: Optional[str] = typer.Option(
        None, help="extract fresh: local endpoint model id (e.g. llama3.2:3b) over --corpus-root"
    ),
    khop_depth: Optional[int] = typer.Option(None, help="local_khop expansion radius (default 2)"),
) -> None:
    """Build the M6 GraphRAG store from the M4.4 extraction (nodes/edges + communities).

    Source precedence: --bundle, else --extraction + --corpus-root, else fresh extraction over
    --corpus-root via a local endpoint (--extract-model). Writes node/edge JSONL + meta under the
    config's graph dir; select it at eval time with `--retrieval-backend graph`.
    """
    from llb.graph.store import GraphStore

    cfg = load_config(config, corpus_root=corpus_root, graph_khop_depth=khop_depth)
    extractions, docs, ontology = _resolve_graph_inputs(cfg, bundle, extraction, extract_model)
    store = GraphStore.build(extractions, docs, ontology, khop_depth=cfg.graph_khop_depth)
    store.save(cfg.graph_dir())
    typer.echo(
        f"[build-graph] {store.meta['n_nodes']} nodes, {store.meta['n_edges']} edges, "
        f"{store.meta['n_communities']} communities -> {cfg.graph_dir()}"
    )


def _resolve_graph_inputs(
    cfg: "RunConfig",
    bundle: Optional[Path],
    extraction: Optional[Path],
    extract_model: Optional[str],
) -> "tuple[list[DocExtraction], list[DocRecord], OntologyCandidate | None]":
    """Load (extractions, docs, ontology) from a bundle, explicit paths, or fresh extraction."""
    from llb.graph.ingest import load_bundle, load_extractions
    from llb.prep.ontology.inventory import inventory_corpus

    if bundle is not None:
        return load_bundle(bundle)
    if extraction is not None:
        docs = inventory_corpus(cfg.corpus_root)
        return load_extractions(extraction), docs, None
    if extract_model is not None:
        from llb.prep.frontier import ProvenanceLog
        from llb.prep.ontology.endpoint import ENDPOINT_LOCAL, EndpointConfig, build_complete
        from llb.prep.ontology.extract import LLMExtractionAdapter, extract_corpus

        docs = inventory_corpus(cfg.corpus_root)
        endpoint = EndpointConfig(kind=ENDPOINT_LOCAL, model=extract_model)
        complete = build_complete(endpoint, ProvenanceLog())
        return extract_corpus(docs, LLMExtractionAdapter(complete)), docs, None
    typer.echo(
        "[error] build-graph needs one of: --bundle, --extraction (+ --corpus-root), "
        "or --extract-model (+ --corpus-root)",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("validate-retrieval")
def validate_retrieval(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, help="recall@k cutoff (Premise 4 gate is recall@10 >= 0.8)"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    retrieval_backend: Optional[str] = typer.Option(None, help="faiss | graph (M6)"),
    retrieval_strategy: Optional[str] = typer.Option(
        None, help="graph strategy: local_khop | global_community"
    ),
) -> None:
    """Score the configured backend's retrieval over the gold set (does not rank models)."""
    from llb.executor.cases import spans_as_dicts
    from llb.executor.runner import _load_store
    from llb.goldset.schema import load_goldset
    from llb.rag import retrieval

    cfg = load_config(
        config,
        goldset_path=goldset,
        retrieval_backend=retrieval_backend,
        retrieval_strategy=retrieval_strategy,
    )
    store = _load_store(cfg)
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    pairs = [(store.retrieve(it.question, k), spans_as_dicts(it)) for it in items]
    report = retrieval.evaluate_retrieval(pairs, k)
    gate = "PASS" if report["recall_at_k"] >= 0.8 else "BELOW 0.8 (retrieval is the bottleneck)"
    typer.echo(
        f"[validate-retrieval] n={report['n']} recall@{k}={report['recall_at_k']:.3f} "
        f"mrr={report['mrr']:.3f} -> {gate}"
    )
