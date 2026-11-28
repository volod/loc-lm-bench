"""Run the book reading against one pinned graph and FAISS index."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.artifacts.retrieval_graph.opaque import content_digest
from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.core.fsutil import atomic_write_text
from llb.core.store_generations import resolve_store_dir
from llb.executor.runner_retrieval import _load_vector_store
from llb.graph.book_inputs import load_book_sources
from llb.graph.book_store import BOOK_GRAPH_DIR, book_graph_strategies, load_book_store
from llb.graph.constants import STRATEGY_LOCAL_KHOP
from llb.graph.store import GraphStore
from llb.rag.book_comparison.graph_reading import (
    PRIMARY_METRIC,
    GraphQuestion,
    ItemReading,
    book_graph_questions,
    graph_slices,
    read_book_graph,
    replay_item,
    summarize_graph_slices,
)
from llb.rag.book_comparison.inputs import load_verified_book_gold
from llb.rag.book_comparison.metrics import ReadingSettings, declared_settings
from llb.rag.comparison.models import Retriever
from llb.rag.duplicates.collapse import collapse_is_lossless
from llb.rag.fusion_evidence.rows import build_sweep_rows
from llb.rag.noise_floor.measure import (
    CANDIDATE_DEPTH_FACTOR,
    DEFAULT_REPLICATES,
    measure_noise_floor,
)
from llb.rag.vector_store.layout import META_FILE
from llb.rag.vector_store.vector_index import RAG_BACKEND_FAISS

METHOD = "compare-book-reference-graph"


@dataclass(frozen=True)
class GraphReadingOptions:
    noise_floor: bool = False
    noise_floor_replicates: int = DEFAULT_REPLICATES
    split: str | None = None
    out_dir: Path | None = None


def run_book_graph_reading(
    cfg: RunConfig, settings: ReadingSettings, options: GraphReadingOptions
) -> tuple[Path, dict[str, dict[str, object]]]:
    """Bind both stores, the projection and verified gold before reading any question."""
    _, stamp = new_run_timestamp()
    target = options.out_dir or cfg.data_dir / METHOD / stamp
    index_dir = resolve_store_dir(cfg.index_dir(), META_FILE)
    if any(
        target.resolve().is_relative_to(path.resolve())
        for path in (
            cfg.corpus_root,
            index_dir,
            cfg.graph_dir(),
            cfg.goldset_path,
        )
    ):
        raise ValueError("comparison output must be outside the projection, gold and stores")
    paths = {"projection": cfg.corpus_root, "goldset": cfg.goldset_path, "index": index_dir}
    digests = {name: content_digest(path) for name, path in paths.items()}
    sources = load_book_sources(cfg.corpus_root)
    docs = {doc_id: doc.source for doc_id, doc in sources.documents.items()}
    gold = load_verified_book_gold(cfg.goldset_path, cfg.corpus_root, docs)
    selected = [item for item in gold if options.split is None or item.split == options.split]
    if not selected:
        raise ValueError("the gold set selection is empty")
    items = book_graph_questions(selected, sources)

    graph_path, graph = load_book_store(
        cfg.graph_dir() / BOOK_GRAPH_DIR, digests["projection"], STRATEGY_LOCAL_KHOP
    )
    paths["graph"] = graph_path
    digests["graph"] = content_digest(graph_path)
    vector = _load_vector_store(cfg, index_dir)
    _check_index_settings(cfg, vector.meta)
    lanes = _replay_lanes(cfg, vector, graph, graph_path, items, settings.k, options.noise_floor)
    readings = read_book_graph(lanes, items, docs, settings.k)
    slices = _read_slices(readings, lanes, settings, options)

    if any(content_digest(path) != digests[name] for name, path in paths.items()):
        raise ValueError("book reference comparison inputs changed during retrieval")
    report = {
        "source_pdf_sha256": sources.source_sha256,
        "inputs": {
            name: {"path": str(path), "digest": digests[name]} for name, path in paths.items()
        },
        "settings": _settings(cfg, settings, options, graph),
        "index": vector.meta,
        "graph": graph.meta,
        "slices": slices,
        "items": [replay_item(reading) for reading in readings],
    }
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target / "run_config.json", json.dumps(cfg.fingerprint(), indent=2) + "\n")
    atomic_write_text(target / "comparison.json", json.dumps(report, indent=2) + "\n")
    return target, slices


def _replay_lanes(
    cfg: RunConfig,
    vector: Retriever,
    graph: GraphStore,
    graph_path: Path,
    items: list[GraphQuestion],
    k: int,
    noise_floor: bool,
) -> dict[str, Retriever]:
    graphs: dict[str, Retriever] = {STRATEGY_LOCAL_KHOP: graph}
    for strategy in book_graph_strategies()[1:]:
        graphs[strategy] = GraphStore.load(graph_path, strategy=strategy)
    return build_sweep_rows(
        vector,
        graphs,
        [item.question for item in items],
        k,
        weights=(cfg.graph_weight,),
        candidates=(cfg.graph_fusion_candidates,),
        identities=(cfg.graph_fusion_span_identity,),
        merge_ratios=(cfg.graph_fusion_span_merge_ratio,),
        pool_depth=CANDIDATE_DEPTH_FACTOR * k if noise_floor else 0,
    )


def _read_slices(
    readings: list[ItemReading],
    lanes: dict[str, Retriever],
    settings: ReadingSettings,
    options: GraphReadingOptions,
) -> dict[str, dict[str, object]]:
    groups = graph_slices(readings)
    slices = summarize_graph_slices(groups, settings)
    if options.noise_floor:
        for name, group in groups.items():
            if group:
                pairs = [(row.item.question, row.item.spans) for row in group]
                slices[name]["noise_floor"] = measure_noise_floor(
                    lanes, pairs, settings.k, replicates=options.noise_floor_replicates
                )
    return slices


def _settings(
    cfg: RunConfig, settings: ReadingSettings, options: GraphReadingOptions, graph: GraphStore
) -> dict[str, object]:
    return {
        **declared_settings(settings),
        "primary_metric": PRIMARY_METRIC,
        "split": options.split,
        "khop_depth": graph.khop_depth,
        "graph_weight": cfg.graph_weight,
        "graph_fusion_candidates": cfg.graph_fusion_candidates,
        "graph_fusion_span_identity": cfg.graph_fusion_span_identity,
        "graph_fusion_span_merge_ratio": cfg.graph_fusion_span_merge_ratio,
        "fusion_weight": cfg.fusion_weight,
        "fusion_candidates": cfg.fusion_candidates,
        "ppr": "included" if "ppr" in book_graph_strategies() else "unavailable",
    }


def _check_index_settings(cfg: RunConfig, meta: dict[str, Any]) -> None:
    expected = {
        "backend": RAG_BACKEND_FAISS,
        "strategy": cfg.strategy,
        "size": cfg.chunk_size,
        "overlap": cfg.chunk_overlap,
        "child_size": cfg.child_chunk_size,
        "mode": cfg.retrieval_mode,
        "duplicate_tier": cfg.duplicate_tier,
        "collapse_duplicates": collapse_is_lossless(cfg.strategy),
    }
    if (
        cfg.retrieval_mode == "hybrid"
        and meta.get("lexical", {}).get("lemmatize") != cfg.lexical_lemmas
    ):
        raise ValueError("index lemmatization differs from the book configuration")
    if differing := [name for name, value in expected.items() if meta.get(name) != value]:
        raise ValueError(f"index differs from the book configuration ({', '.join(differing)})")
