"""RAG index build and retrieval validation commands."""

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config

if TYPE_CHECKING:
    from llb.config import RunConfig
    from llb.prep.frontier import LLMComplete
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
    vector_store: str = typer.Option(
        "faiss",
        help="vector backend behind the RAG-store seam: faiss (default) | chroma ([rag-chroma]) | "
        "qdrant ([rag-qdrant]) | lancedb ([rag-lancedb]); the backend is recorded in the store meta",
    ),
) -> None:
    """Chunk + embed the corpus into a RAG store (FAISS by default) under the index dir."""
    from llb.rag.store import RagStore
    from llb.rag.vector_index import RAG_BACKENDS

    if vector_store not in RAG_BACKENDS:
        typer.echo(
            f"[error] unknown --vector-store '{vector_store}'; choose one of "
            f"{', '.join(RAG_BACKENDS)}",
            err=True,
        )
        raise typer.Exit(code=2)
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

    store = RagStore.build(
        cfg.corpus_root,
        cfg.strategy,
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.embedding_model,
        mode=cfg.retrieval_mode,
        child_size=cfg.child_chunk_size,
        vector_store=vector_store,
    )
    store.save(cfg.index_dir())
    parents = f", {store.meta['n_parents']} parents" if store.meta["n_parents"] else ""
    typer.echo(
        f"[build-index] {store.meta['n_indexed']} indexed chunks{parents} "
        f"({cfg.strategy}/{cfg.retrieval_mode}, {vector_store}, dim {store.meta['dim']}) "
        f"-> {cfg.index_dir()}"
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
    extract_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL for fresh extraction (e.g. vLLM http://host:8000/v1)"
    ),
    extract_max_tokens: Optional[int] = typer.Option(
        None, help="per-call output token budget for fresh extraction (raise for reasoning models)"
    ),
    extract_no_think: bool = typer.Option(
        False,
        "--extract-no-think",
        help="disable reasoning (Ollama think=false) so a reasoning model (gemma4) emits JSON directly",
    ),
    khop_depth: Optional[int] = typer.Option(None, help="local_khop expansion radius (default 2)"),
    summarize: bool = typer.Option(
        False,
        "--summarize",
        help="attach diagnostic per-community summaries (needs a local endpoint; never span-scored)",
    ),
    summarize_model: Optional[str] = typer.Option(
        None, help="local endpoint model for --summarize (defaults to --extract-model)"
    ),
) -> None:
    """Build the M6 GraphRAG store from the M4.4 extraction (nodes/edges + communities).

    Source precedence: --bundle, else --extraction + --corpus-root, else fresh extraction over
    --corpus-root via a local endpoint (--extract-model). Writes node/edge JSONL + meta under the
    config's graph dir; select it at eval time with `--retrieval-backend graph`. With --summarize it
    also writes the tagged-diagnostic community summaries (recorded, never returned by retrieval).
    """
    from llb.graph.store import GraphStore

    cfg = load_config(config, corpus_root=corpus_root, graph_khop_depth=khop_depth)
    think = False if extract_no_think else None
    extractions, docs, ontology = _resolve_graph_inputs(
        cfg,
        bundle,
        extraction,
        extract_model,
        base_url=extract_base_url,
        max_tokens=extract_max_tokens,
        think=think,
    )
    store = GraphStore.build(extractions, docs, ontology, khop_depth=cfg.graph_khop_depth)
    summary_note = ""
    if summarize:
        store.community_summaries = _summarize_graph(
            store,
            summarize_model or extract_model,
            base_url=extract_base_url,
            max_tokens=extract_max_tokens,
            think=think,
        )
        summary_note = f", {len(store.community_summaries)} community summaries"
    store.save(cfg.graph_dir())
    typer.echo(
        f"[build-graph] {store.meta['n_nodes']} nodes, {store.meta['n_edges']} edges, "
        f"{store.meta['n_communities']} communities{summary_note} -> {cfg.graph_dir()}"
    )


def _local_complete(
    model: str,
    *,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
    think: Optional[bool] = None,
) -> "LLMComplete":
    """Build the injectable M4.4 local-endpoint completion callable for `model`.

    `base_url` points at the OpenAI-compatible server (defaults to local Ollama; pass a vLLM
    `http://host:port/v1` to serve a quantized HF checkpoint). `max_tokens` raises the per-call
    output budget and `think=False` disables a reasoning model's hidden thinking (Ollama native),
    so a calibrated reasoning model emits JSON directly.
    """
    from llb.prep.frontier import ProvenanceLog
    from llb.prep.ontology.endpoint import ENDPOINT_LOCAL, EndpointConfig, build_complete

    cfg = EndpointConfig(kind=ENDPOINT_LOCAL, model=model, think=think)
    if base_url is not None:
        cfg = replace(cfg, base_url=base_url)
    if max_tokens is not None:
        cfg = replace(cfg, max_tokens=max_tokens)
    return build_complete(cfg, ProvenanceLog())


def _summarize_graph(
    store: object,
    model: Optional[str],
    *,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
    think: Optional[bool] = None,
) -> dict[str, str]:
    """Generate the tagged-diagnostic community summaries for `store` via a local endpoint."""
    if not model:
        typer.echo("[error] --summarize needs --summarize-model (or --extract-model)", err=True)
        raise typer.Exit(code=2)
    from llb.graph.store import GraphStore
    from llb.graph.summary import summarize_communities

    assert isinstance(store, GraphStore)  # narrow for mypy; callers pass a built store
    complete = _local_complete(model, base_url=base_url, max_tokens=max_tokens, think=think)
    return summarize_communities(store.graph, complete)


def _resolve_graph_inputs(
    cfg: "RunConfig",
    bundle: Optional[Path],
    extraction: Optional[Path],
    extract_model: Optional[str],
    *,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
    think: Optional[bool] = None,
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
        from llb.prep.ontology.extract import LLMExtractionAdapter, extract_corpus

        docs = inventory_corpus(cfg.corpus_root)
        complete = _local_complete(
            extract_model, base_url=base_url, max_tokens=max_tokens, think=think
        )
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


@app.command("compare-retrieval")
def compare_retrieval_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, help="recall@k / MRR cutoff"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    out: Optional[Path] = typer.Option(None, help="write the JSON comparison report here"),
) -> None:
    """Compare FAISS vs graph/local_khop vs graph/global_community retrieval on one gold set (M6).

    Scores each BUILT backend's recall@k / MRR on the SAME items by the source-span metric (a
    backend whose store is not built is skipped). Quantifies when the graph paths beat flat vector
    retrieval; answer-quality comparison rides `run-eval --retrieval-backend ...` (it needs a model).
    """
    import json

    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag.compare import compare_retrieval, format_comparison, load_compare_stores

    cfg = load_config(config, goldset_path=goldset)
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    stores = load_compare_stores(cfg)
    if not stores:
        typer.echo(
            "[error] no retrieval backend is built (run build-index / build-graph)", err=True
        )
        raise typer.Exit(code=2)
    report = compare_retrieval(stores, [(it.question, spans_as_dicts(it)) for it in items], k)
    typer.echo(format_comparison(report))
    if out is not None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[compare-retrieval] wrote report -> {out}")


@app.command("compare-vector-stores")
def compare_vector_stores_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    backends: str = typer.Option(
        "faiss,chroma,qdrant,lancedb",
        help="comma-separated vector backends to compare (each over the SAME corpus + embedder)",
    ),
    k: int = typer.Option(10, help="recall@k / MRR cutoff"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    out: Optional[Path] = typer.Option(None, help="write the JSON comparison report here"),
) -> None:
    """M7.4: compare vector-store backends (FAISS vs Chroma/Qdrant/LanceDB) by the source-span metric.

    Builds the SAME corpus under each backend with the SAME chunking + pinned embedder, then scores
    recall@k / MRR on the gold set -- the model-independent retrieval gate before a backend's runs
    can be compared to FAISS. Each non-FAISS backend needs its optional extra installed."""
    import json

    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag.compare import build_vector_store_comparison, compare_retrieval, format_comparison

    cfg = load_config(config, goldset_path=goldset)
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    selected = [b.strip() for b in backends.split(",") if b.strip()]
    stores = build_vector_store_comparison(cfg, selected)
    report = compare_retrieval(stores, [(it.question, spans_as_dicts(it)) for it in items], k)
    typer.echo(format_comparison(report))
    if out is not None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[compare-vector-stores] wrote report -> {out}")
