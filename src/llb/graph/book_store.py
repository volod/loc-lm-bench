"""Publish and load the structural book graph independently of the entity store."""

import shutil
from pathlib import Path

from llb.artifacts.retrieval_graph.opaque import content_digest
from llb.bench.common import new_run_timestamp
from llb.graph.book_inputs import load_book_sources
from llb.graph.book_references import build_book_reference_graph
from llb.graph.constants import BACKEND_GRAPH, META_FILE, STRATEGY_LOCAL_KHOP
from llb.graph.store import GraphStore
from llb.core.store_generations import new_generation_paths, publish_generation, resolve_store_dir
from llb.prep.corpus.fingerprints import corpus_doc_fingerprints

BOOK_GRAPH_DIR = "book-references"
BOOK_KHOP_DEPTH = 1


def build_book_store(corpus: Path, base_dir: Path, *, khop_depth: int) -> tuple[Path, GraphStore]:
    if khop_depth < 1:
        raise ValueError("book graph hop depth must be at least one")
    if base_dir.resolve().is_relative_to(corpus.resolve()):
        raise ValueError("book graph output must be outside its input corpus")
    digest = content_digest(corpus)
    sources = load_book_sources(corpus)
    graph = build_book_reference_graph(sources)
    store = GraphStore(
        graph,
        {
            "backend": BACKEND_GRAPH,
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "n_communities": 0,
            "n_documents": len(sources.documents),
            "khop_depth": khop_depth,
            "doc_fingerprints": corpus_doc_fingerprints(corpus),
            "corpus_digest": digest,
        },
        khop_depth=khop_depth,
    )
    _, timestamp = new_run_timestamp()
    staging, final = new_generation_paths(base_dir, timestamp)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        store.save(staging)
        if content_digest(corpus) != digest:
            raise ValueError("book projection changed during graph build")
        return publish_generation(staging, final), store
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_book_store(base_dir: Path, corpus_digest: str, strategy: str) -> tuple[Path, GraphStore]:
    generation = resolve_store_dir(base_dir, META_FILE)
    store = GraphStore.load(generation, strategy=strategy)
    if store.meta.get("corpus_digest") != corpus_digest:
        raise ValueError("stale book graph: rebuild it over this projection with make build-graph")
    return generation, store


def book_graph_strategies() -> tuple[str, ...]:
    from llb.graph.constants import STRATEGIES

    return (STRATEGY_LOCAL_KHOP, "ppr") if "ppr" in STRATEGIES else (STRATEGY_LOCAL_KHOP,)
