"""Eval inputs for the runner: gold items, retrieval store, default per-case runner fn, the opt-in
query-prep lane, and the abstention probe. `runner_backend.py` owns the backend lifecycle.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.eval import graph as eval_graph
from llb.executor.cases import spans_as_dicts
from llb.goldset.schema import GoldItem, load_goldset

if TYPE_CHECKING:
    from llb.eval.insufficient_context import InsufficientContextReport
    from llb.executor.cases import ScoreOptions

RagState = eval_graph.RagState
_LOG = logging.getLogger(__name__)


def _load_eval_items(config: RunConfig, split: str, limit: int | None) -> list[GoldItem]:
    if not config.goldset_path.exists():
        raise SystemExit(
            f"gold set not found: {config.goldset_path}\n"
            "  use the committed fixture with --goldset "
            "samples/goldsets/ua_squad_postedited_v1/goldset.jsonl,\n"
            "  or create unverified development material with `make ingest-uk-squad`."
        )
    items = [
        item for item in load_goldset(config.goldset_path) if item.split == split and item.verified
    ]
    items.sort(key=lambda it: it.id)
    return items[:limit] if limit is not None else items


def _select_eval_items(
    config: RunConfig,
    items: list[GoldItem] | None,
    split: str,
    limit: int | None,
) -> list[GoldItem]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if items is None:
        return _load_eval_items(config, split, limit)
    selected = sorted(
        (item for item in items if item.split == split and item.verified),
        key=lambda item: item.id,
    )
    return selected[:limit] if limit is not None else selected


def _default_runner_fn(
    config: RunConfig, store: Any, launcher: BackendLauncher, prompt_package: Any | None = None
) -> Callable[[GoldItem], RagState]:
    chunk_filter = None
    if config.acl_label is not None:
        from llb.rag.filters import metadata_filter

        chunk_filter = metadata_filter(acl_label=config.acl_label)
    app = eval_graph.build_rag_graph(
        store,
        launcher,
        config.top_k,
        config.max_tokens,
        config.temperature,
        config.request_timeout_s,
        prompt_package=prompt_package,
        context_order=config.context_order,
        query_prep=build_query_prep(config, store, launcher),
        chunk_filter=chunk_filter,
        cited=config.cited_answers,
    )

    def run(item: GoldItem) -> RagState:
        return eval_graph.run_case(app, item.question, spans_as_dicts(item))

    return run


def _score_options(config: RunConfig) -> "ScoreOptions":
    """The opt-in answer-side scoring toggles for this run (groundedness-citation-metrics)."""
    from llb.executor.cases import ScoreOptions

    return ScoreOptions(
        score_groundedness=config.score_groundedness,
        cited_answers=config.cited_answers,
        context_order=config.context_order,
    )


def _maybe_run_probes(
    config: RunConfig, items: list[GoldItem], store: Any, backend: Any
) -> "InsufficientContextReport | None":
    """Run the insufficient-context abstention probe over a seeded sample, if configured.

    The gold evidence is excluded from retrieval for each probed item; correct behavior is an
    explicit abstention. Probe rows are scored separately and never enter the correctness batch."""
    if config.insufficient_context_probes <= 0:
        return None
    from llb.eval.insufficient_context import run_insufficient_context_probe

    def chat(messages: Any) -> tuple[str, str | None]:
        result = backend.chat(
            messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.request_timeout_s,
        )
        return result.text or "", result.error

    return run_insufficient_context_probe(
        items,
        store,
        chat,
        model=config.model,
        backend=config.backend,
        k=config.top_k,
        n=config.insufficient_context_probes,
        seed=config.seed,
        cited=config.cited_answers,
        context_order=config.context_order,
    )


def _launcher_rewriter(config: RunConfig, launcher: Any) -> Callable[[str], str]:
    """Local-LLM query rewriter over the run's backend endpoint seam (uk-query-processing)."""
    from llb.prompts import render_chat

    def rewrite(query: str) -> str:
        messages = render_chat("eval.rag.query_rewrite", {"query": query})
        result = launcher.chat(
            messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.request_timeout_s,
        )
        return result.text or ""

    return rewrite


def build_query_prep(config: RunConfig, store: Any, launcher: Any | None) -> Any | None:
    """Build the opt-in query-side lane for this run, resolving each step's dependency.

    Returns None when no steps are configured (the lane is an exact no-op). The typo step reads
    the corpus vocabulary from the loaded store's chunks; the glossary step loads
    `config.query_glossary_path`; the rewrite step wraps the backend launcher. Missing
    dependencies raise a clear SystemExit rather than a bare error mid-run."""
    from llb.rag.query_prep.base import STEP_GLOSSARY, STEP_REWRITE, STEP_TYPOS
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
    if STEP_REWRITE in steps:
        if launcher is None:
            raise SystemExit("[run-eval] query_prep 'rewrite' step needs a backend launcher")
        rewriter = _launcher_rewriter(config, launcher)
    try:
        return QueryPrep.build(
            steps,
            vocabulary=vocabulary,
            glossary=glossary,
            rewriter=rewriter,
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
    """Load the configured retrieval store: the GraphRAG backend (GraphRAG backend) or the default FAISS store.

    Both expose the same `.retrieve(question, k) -> list[ChunkRecord]` seam, so the eval graph,
    scoring, isolation, and board are unchanged regardless of backend. With `config.reranker`
    set, the loaded store is wrapped in the cross-encoder rerank stage (rerank-context-order);
    the wrapper honors the same retrieve seam, so every backend gains reranking identically."""
    from llb.rag.rerank import maybe_wrap_reranker

    if config.retrieval_backend == "graph":
        from llb.graph.store import GraphStore

        return maybe_wrap_reranker(
            GraphStore.load(
                config.graph_dir(),
                strategy=config.retrieval_strategy,
                khop_depth=config.graph_khop_depth,
            ),
            config,
        )
    from llb.rag.store import MODE_HYBRID, RagStore, stale_store_message, store_embedder_mismatch

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
    return maybe_wrap_reranker(store, config)
