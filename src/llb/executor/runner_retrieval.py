"""Focused runner retrieval implementation."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from llb.core.config import RunConfig

if TYPE_CHECKING:
    pass
_LOG = logging.getLogger(__name__)


def _launcher_generator(config: RunConfig, launcher: Any, prompt_id: str) -> Callable[[str], str]:
    """Local-LLM query generator over the run's backend endpoint seam."""
    from llb.prompts.registry import render_chat

    cache: dict[str, str] = {}

    def generate(query: str) -> str:
        if query in cache:
            return cache[query]
        messages = render_chat(prompt_id, {"query": query})
        result = launcher.chat(
            messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.request_timeout_s,
        )
        generated = result.text or ""
        cache[query] = generated
        return generated

    return generate


def build_query_prep(config: RunConfig, store: Any, launcher: Any | None) -> Any | None:
    """Build the opt-in query-side lane for this run, resolving each step's dependency.

    Returns None when no steps are configured (the lane is an exact no-op). The typo step reads
    the corpus vocabulary from the loaded store's chunks; the glossary step loads
    `config.query_glossary_path`; the rewrite step wraps the backend launcher. Missing
    dependencies raise a clear SystemExit rather than a bare error mid-run."""
    from llb.rag.query_prep.base import (
        STEP_DECOMPOSE,
        STEP_GLOSSARY,
        STEP_HYDE,
        STEP_REWRITE,
        STEP_TYPOS,
    )
    from llb.rag.query_prep.pipeline import QueryPrep
    from llb.rag.query_prep.typos import build_vocabulary

    steps = list(config.query_prep)
    if not steps:
        return None
    vocabulary = None
    known_word = None
    if STEP_TYPOS in steps:
        chunks = getattr(store, "chunks", None) or []
        vocabulary = build_vocabulary(str(chunk.get("text", "")) for chunk in chunks)
        if config.query_prep_typo_guard:
            from llb.rag.lexical import load_uk_word_probe

            known_word = load_uk_word_probe()
    glossary = _load_query_glossary(config) if STEP_GLOSSARY in steps else None
    rewriter = None
    model_steps = {STEP_REWRITE, STEP_HYDE, STEP_DECOMPOSE}.intersection(steps)
    if model_steps:
        if launcher is None:
            joined = ",".join(sorted(model_steps))
            raise SystemExit(f"[run-eval] query_prep '{joined}' needs a backend launcher")
    rewriter = (
        _launcher_generator(config, launcher, "eval.rag.query_rewrite")
        if STEP_REWRITE in steps
        else None
    )
    hypothesizer = (
        _launcher_generator(config, launcher, "eval.rag.query_hyde") if STEP_HYDE in steps else None
    )
    decomposer = (
        _launcher_generator(config, launcher, "eval.rag.query_decompose")
        if STEP_DECOMPOSE in steps
        else None
    )
    try:
        return QueryPrep.build(
            steps,
            vocabulary=vocabulary,
            glossary=glossary,
            rewriter=rewriter,
            hypothesizer=hypothesizer,
            decomposer=decomposer,
            known_word=known_word,
        )
    except ValueError as exc:
        raise SystemExit(f"[run-eval] invalid query_prep: {exc}") from None


def _load_query_glossary(config: RunConfig) -> Any:
    """The configured query glossary, with clear SystemExit errors for missing configuration."""
    from llb.rag.query_prep.glossary import Glossary

    if config.query_glossary_path is None:
        raise SystemExit(
            "[run-eval] query_prep 'glossary' step needs query_glossary_path "
            "(build one with `llb build-query-glossary`)."
        )
    if not Path(config.query_glossary_path).is_file():
        raise SystemExit(f"[run-eval] query glossary not found: {config.query_glossary_path}")
    return Glossary.load(config.query_glossary_path)


def _load_store(config: RunConfig) -> Any:
    """Load the configured vector, graph, or graph-vector fused retrieval store.

    Both expose the same `.retrieve(question, k) -> list[ChunkRecord]` seam, so the eval graph,
    scoring, isolation, and board are unchanged regardless of backend. With `config.reranker`
    set, the loaded store is wrapped in the cross-encoder rerank stage (rerank-context-order);
    the wrapper honors the same retrieve seam, so every backend gains reranking identically."""
    from llb.rag.rerank import maybe_wrap_reranker

    graph = None
    if config.retrieval_backend in {"graph", "fused"}:
        graph = _load_graph_store(config)
        if config.retrieval_backend == "graph":
            return maybe_wrap_reranker(graph, config)
    vector = _load_vector_store(config)
    if config.retrieval_backend == "fused":
        from llb.rag.fusion import FusedRetriever

        assert graph is not None
        fused = FusedRetriever(
            vector,
            graph,
            config.graph_weight,
            config.graph_fusion_candidates,
            config.graph_fusion_span_identity,
        )
        return maybe_wrap_reranker(fused, config)
    return maybe_wrap_reranker(vector, config)


def _load_graph_store(config: RunConfig) -> Any:
    """Load the configured span-preserving graph strategy."""
    from llb.graph.store import GraphStore

    return GraphStore.load(
        config.graph_dir(),
        strategy=config.retrieval_strategy,
        khop_depth=config.graph_khop_depth,
    )


def _load_vector_store(config: RunConfig) -> Any:
    """Load and validate the vector lane, including optional dense/BM25 fusion."""
    from llb.rag.store import RagStore
    from llb.rag.store_build import MODE_HYBRID
    from llb.rag.store_validation import stale_store_message, store_embedder_mismatch

    store = RagStore.load(config.index_dir())
    stale = stale_store_message(store.meta, config.corpus_root, config.index_dir())
    if stale is not None:
        raise SystemExit(stale)
    built = store_embedder_mismatch(store.meta, config.embedding_model)
    if built is not None:
        raise SystemExit(
            f"[run-eval] embedder mismatch: the store at {config.index_dir()} was built with "
            f"'{built}' but config.embedding_model is '{config.embedding_model}'. Rebuild the "
            f"index (build-index --embedding-model {config.embedding_model}) or set the config to "
            f"match; a store is embedded and queried by one encoder, so they must agree."
        )
    if config.retrieval_mode == MODE_HYBRID:
        if getattr(store, "lexical", None) is None:
            raise SystemExit(
                f"[run-eval] --retrieval-mode hybrid needs a lexical index, but the store at "
                f"{config.index_dir()} was built '{store.meta.get('mode')}' (dense-only). "
                f"Rebuild it with `build-index --retrieval-mode hybrid`."
            )
        store.fusion_weight = config.fusion_weight
        store.fusion_candidates = config.fusion_candidates
    elif getattr(store, "lexical", None) is not None:
        # A hybrid store can always serve dense-only: drop the lexical side for this run.
        _LOG.info(
            "[run-eval] retrieval_mode=%s over a hybrid store; lexical fusion disabled",
            config.retrieval_mode,
        )
        store.lexical = None
    return store
