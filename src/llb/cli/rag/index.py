"""RAG/GraphRAG index build commands (vector index + graph store)."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config

if TYPE_CHECKING:
    pass


@app.command("build-index")
def build_index(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    corpus_root: Optional[Path] = typer.Option(None, help="corpus directory to chunk"),
    strategy: Optional[str] = typer.Option(
        None, help="fixed | sentence | recursive | markdown | semantic | page | heading | late"
    ),
    size: Optional[int] = typer.Option(None, help="chunk size (chars)"),
    overlap: Optional[int] = typer.Option(None, help="chunk overlap (chars)"),
    embedding_model: Optional[str] = typer.Option(None, help="pinned embedding model"),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        "--retrieval-mode",
        help="flat | parent_child | hybrid (hybrid adds a lexical BM25 index beside the vectors)",
    ),
    child_size: Optional[int] = typer.Option(None, help="child chunk size (parent_child mode)"),
    lemmatize: bool = typer.Option(
        False,
        "--lemmatize",
        help="hybrid mode: collapse Ukrainian inflection to lemmas on the LEXICAL side "
        "(pymorphy3, the [lex] extra); stored chunk text is never altered",
    ),
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
        lexical_lemmas=lemmatize or None,
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
        lexical_lemmas=cfg.lexical_lemmas,
    )
    store.save(cfg.index_dir())
    parents = f", {store.meta['n_parents']} parents" if store.meta["n_parents"] else ""
    coverage = store.meta.get("page_annotation_coverage", 0.0)
    pages = f", {coverage:.0%} page-annotated" if coverage else ""
    lexical_meta = store.meta.get("lexical")
    lexical = (
        f", lexical {lexical_meta['n_terms']} terms"
        f"{' (lemmatized)' if lexical_meta['lemmatize'] else ''}"
        if lexical_meta
        else ""
    )
    typer.echo(
        f"[build-index] {store.meta['n_indexed']} indexed chunks{parents} "
        f"({cfg.strategy}/{cfg.retrieval_mode}, {vector_store}, dim {store.meta['dim']}) "
        f"-> {cfg.index_dir()}{pages}{lexical}"
    )
