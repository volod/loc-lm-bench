"""RAG index build and retrieval validation commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config


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


@app.command("validate-retrieval")
def validate_retrieval(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, help="recall@k cutoff (Premise 4 gate is recall@10 >= 0.8)"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
) -> None:
    """Score the pinned embedding's retrieval over the gold set (does not rank models)."""
    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag import retrieval
    from llb.rag.store import RagStore

    cfg = load_config(config, goldset_path=goldset)
    store = RagStore.load(cfg.index_dir())
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
