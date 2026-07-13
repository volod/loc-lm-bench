"""Compare retrieval backends on ONE gold set by the source-span metric (GraphRAG backend residual 3).

Quantifies when the GraphRAG multi-hop / narrative paths beat flat vector retrieval: it runs the
SAME goldset through several backends -- typically `{faiss, graph/local_khop, graph/global_community}`
-- and reports each one's `recall@k` / `MRR` (the model-independent retrieval axis the manifest's
backend + strategy already make comparable). Answer-quality comparison rides the normal
`run-eval --retrieval-backend ...` path (it needs a model); this tool isolates the retrieval signal.

Pure: it takes any object exposing `.retrieve(question, k) -> list[ChunkRecord]` (the RAG-store
seam), so it is unit-tested with fake stores -- no GPU, no FAISS, no DuckDB. Each backend reuses the
one `evaluate_retrieval` span metric, so graph and FAISS score on identical rules.
"""

from typing import Any, Protocol

from typing_extensions import TypedDict

from llb.core.contracts import ChunkRecord, RetrievalMetrics, SourceSpanRecord
from llb.rag.filters import metadata_filter
from llb.rag.retrieval import evaluate_retrieval

# (question, gold source spans) -- the per-item input shared across every compared backend.
CompareItem = tuple[str, list[SourceSpanRecord]]

# Row labels of the hybrid comparison (hybrid-retrieval-uk).
ROW_DENSE = "dense"
ROW_HYBRID = "hybrid"
ROW_HYBRID_LEMMAS = "hybrid+lemmas"
ROW_ORACLE_DOC = "dense+oracle-doc"
# Suffix of the reranked twin row (rerank-context-order): `<label>+rerank` scores the SAME
# store's candidates after the cross-encoder cut, so pre/post-rerank recall@k / MRR compare
# through the one `evaluate_retrieval` metric.
RERANK_ROW_SUFFIX = "+rerank"


class Retriever(Protocol):
    """The RAG-store seam every compared backend implements (FAISS or GraphStore)."""

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]: ...


class ComparisonReport(TypedDict):
    """Per-backend span metrics over one goldset plus the recall winner (None if no backends)."""

    k: int
    n: int
    backends: dict[str, RetrievalMetrics]
    best_recall: str | None


def compare_retrieval(
    stores: dict[str, Retriever], items: list[CompareItem], k: int
) -> ComparisonReport:
    """Score each labeled backend's top-k retrieval over the same items; rank by recall@k."""
    per_backend: dict[str, RetrievalMetrics] = {}
    for label, store in stores.items():
        pairs = [(store.retrieve(question, k), spans) for question, spans in items]
        per_backend[label] = evaluate_retrieval(pairs, k)
    return {
        "k": k,
        "n": len(items),
        "backends": per_backend,
        "best_recall": _best_recall(per_backend),
    }


def _best_recall(per_backend: dict[str, RetrievalMetrics]) -> str | None:
    """Label with the highest recall@k (tie-break: higher MRR, then label order)."""
    if not per_backend:
        return None
    return min(
        per_backend,
        key=lambda label: (
            -per_backend[label]["recall_at_k"],
            -per_backend[label]["mrr"],
            label,
        ),
    )


def add_rerank_rows(
    stores: dict[str, Retriever], scorer: Any, candidates: int
) -> dict[str, Retriever]:
    """Add a `<label>+rerank` twin per compared store (rerank-context-order).

    Each twin wraps the SAME store in the cross-encoder rerank stage (retrieve `candidates`,
    rerank, keep k), so the report shows the pre/post-rerank recall@k / MRR delta per backend.
    The oracle-doc headroom row is skipped (it is a diagnostic bound, not a rankable config).
    `scorer` is the injectable `RerankScorer` (a fake in tests; `CrossEncoderReranker` real).
    """
    from llb.rag.rerank import RerankingRetriever

    out: dict[str, Retriever] = dict(stores)
    for label, store in stores.items():
        if label == ROW_ORACLE_DOC:
            continue
        out[f"{label}{RERANK_ROW_SUFFIX}"] = RerankingRetriever(
            store, scorer, candidates=candidates
        )
    return out


def format_comparison(report: ComparisonReport) -> str:
    """Render an ASCII comparison table (AGENTS.md: ASCII-only, no box-drawing)."""
    backends = report["backends"]
    lines = [f"[compare-retrieval] n={report['n']} k={report['k']}"]
    if not backends:
        lines.append("  (no backends loaded)")
        return "\n".join(lines)
    width = max(len(label) for label in backends)
    lines.append(f"  {'backend'.ljust(width)}   recall@k      mrr")
    for label in sorted(backends):
        metrics = backends[label]
        lines.append(
            f"  {label.ljust(width)}   {metrics['recall_at_k']:8.3f} {metrics['mrr']:8.3f}"
        )
    lines.append(f"  best (recall@k): {report['best_recall']}")
    return "\n".join(lines)


def build_vector_store_comparison(config: Any, backends: list[str]) -> dict[str, Retriever]:
    """Build the SAME corpus under each vector-store backend for a source-span retrieval comparison.

    Every store reuses the config's chunking + PINNED embedder and differs ONLY in the vector
    backend (faiss / chroma / qdrant / lancedb), so `compare_retrieval` isolates the backend's
    effect on recall@k / MRR -- the model-independent gate the platform matrix plan requires before a backend's
    runs can be compared to FAISS. Real path: needs the [rag] embedder + each backend's extra."""
    from llb.rag.store import RagStore
    from llb.rag.vector_index import RAG_BACKENDS

    stores: dict[str, Retriever] = {}
    for backend in backends:
        if backend not in RAG_BACKENDS:
            raise ValueError(
                f"unknown vector store backend: {backend!r}; choose from {RAG_BACKENDS}"
            )
        stores[backend] = RagStore.build(
            config.corpus_root,
            config.strategy,
            config.chunk_size,
            config.chunk_overlap,
            config.embedding_model,
            mode=config.retrieval_mode,
            child_size=config.child_chunk_size,
            vector_store=backend,
            lexical_lemmas=config.lexical_lemmas,
        )
    return stores


def build_chunking_comparison(
    config: Any, strategies: list[str], stores_root: Any = None
) -> dict[str, Retriever]:
    """Build one FAISS store per CHUNKING strategy for a source-span retrieval comparison.

    Every store shares the config's corpus, chunk size/overlap, and PINNED embedder and differs
    ONLY in the chunking strategy, so `compare_retrieval` demonstrates (not assumes) the best
    chunker per corpus. Stores are built in `flat` mode -- parent_child would confound the
    boundary comparison (and `late` refuses it). When `stores_root` is given each store persists
    under `<stores_root>/<strategy>/` for reuse. Real path: needs the `[rag]` extra.
    """
    from pathlib import Path

    from llb.rag.chunking.dispatch import STRATEGIES
    from llb.rag.store import RagStore

    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValueError(
            f"unknown chunking strategy: {unknown[0]!r}; choose from {', '.join(STRATEGIES)}"
        )
    stores: dict[str, Retriever] = {}
    for strategy in strategies:
        store = RagStore.build(
            config.corpus_root,
            strategy,
            config.chunk_size,
            config.chunk_overlap,
            config.embedding_model,
            mode="flat",
        )
        if stores_root is not None:
            store.save(Path(stores_root) / strategy)
        stores[strategy] = store
    return stores


class OracleDocFilter:
    """Diagnostic retriever: restrict candidates to each gold item's own source document(s).

    Wraps a store whose `.retrieve` accepts a `chunk_filter` and scopes every question to the
    doc ids of its labeled spans -- the recall a PERFECT document router would reach. The row
    quantifies routing headroom only; it is never a scoring configuration (the gold doc id is
    an oracle input no real run has).
    """

    def __init__(self, store: Any, items: list[CompareItem]) -> None:
        self._store = store
        self._docs_by_question = {
            question: {span["doc_id"] for span in spans} for question, spans in items
        }

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        doc_ids = self._docs_by_question.get(question)
        if not doc_ids:
            return self._store.retrieve(question, k)  # type: ignore[no-any-return]
        return self._store.retrieve(  # type: ignore[no-any-return]
            question, k, chunk_filter=metadata_filter(doc_ids=doc_ids)
        )


def build_hybrid_comparison(
    config: Any, items: list[CompareItem], stores_root: Any = None
) -> dict[str, Retriever]:
    """Build the hybrid-retrieval row set over ONE embedded corpus (hybrid-retrieval-uk).

    One hybrid store is built (corpus embedded once); the rows share its chunks + dense index
    and differ only in the query path: `dense` (fusion disabled), `hybrid` (BM25 + weighted
    RRF with the config's fusion knobs), `hybrid+lemmas` (a second lexical index with
    Ukrainian lemmatization; skipped with a log line when the `[lex]` extra is absent), and
    `dense+oracle-doc` (candidates restricted to each gold item's source doc -- the recall
    headroom a perfect document router would buy). When `stores_root` is given the hybrid
    store persists under `<stores_root>/hybrid/`. Real path: needs the `[rag]` extra.
    """
    import logging
    from pathlib import Path

    from llb.rag.lexical import LexicalIndex, load_uk_lemmatizer
    from llb.rag.store import MODE_HYBRID, RagStore

    log = logging.getLogger(__name__)
    hybrid = RagStore.build(
        config.corpus_root,
        config.strategy,
        config.chunk_size,
        config.chunk_overlap,
        config.embedding_model,
        mode=MODE_HYBRID,
    )
    hybrid.fusion_weight = config.fusion_weight
    hybrid.fusion_candidates = config.fusion_candidates
    if stores_root is not None:
        hybrid.save(Path(stores_root) / MODE_HYBRID)

    dense_meta = dict(hybrid.meta)
    dense_meta.pop("lexical", None)
    dense_meta["mode"] = "flat"
    dense = RagStore(hybrid.chunks, hybrid.index, hybrid.embedder, dense_meta)  # type: ignore[arg-type]

    stores: dict[str, Retriever] = {
        ROW_DENSE: dense,
        ROW_HYBRID: hybrid,
        ROW_ORACLE_DOC: OracleDocFilter(dense, items),
    }
    try:
        lemmatizer = load_uk_lemmatizer()
    except SystemExit as exc:
        log.warning("[compare-retrieval] skip %s: %s", ROW_HYBRID_LEMMAS, exc)
        return stores
    lemma_index = LexicalIndex.build(
        [c["text"] for c in hybrid.chunks], lemmatize=True, lemmatizer=lemmatizer
    )
    lemma_meta = dict(hybrid.meta)
    lemma_meta["lexical"] = {"lemmatize": True, "n_terms": len(lemma_index.postings)}
    lemma_store = RagStore(
        hybrid.chunks,
        hybrid.index,
        hybrid.embedder,
        lemma_meta,  # type: ignore[arg-type]
        lexical=lemma_index,
    )
    lemma_store.fusion_weight = config.fusion_weight
    lemma_store.fusion_candidates = config.fusion_candidates
    stores[ROW_HYBRID_LEMMAS] = lemma_store
    return stores


def load_compare_stores(config: Any) -> dict[str, Retriever]:
    """Load the three standard backends for `config`, skipping any whose store is not built.

    Returns `{faiss, graph/local_khop, graph/global_community}` -> store. A backend that has no
    built store on disk is skipped with a log line, so the comparison runs over whatever is present.
    """
    import logging

    from llb.executor.runner_setup import _load_store
    from llb.graph.constants import (
        BACKEND_GRAPH,
        STRATEGY_GLOBAL_COMMUNITY,
        STRATEGY_LOCAL_KHOP,
    )

    log = logging.getLogger(__name__)
    plans = {
        "faiss": config.with_overrides(retrieval_backend="faiss"),
        f"{BACKEND_GRAPH}/{STRATEGY_LOCAL_KHOP}": config.with_overrides(
            retrieval_backend=BACKEND_GRAPH, retrieval_strategy=STRATEGY_LOCAL_KHOP
        ),
        f"{BACKEND_GRAPH}/{STRATEGY_GLOBAL_COMMUNITY}": config.with_overrides(
            retrieval_backend=BACKEND_GRAPH, retrieval_strategy=STRATEGY_GLOBAL_COMMUNITY
        ),
    }
    stores: dict[str, Retriever] = {}
    for label, plan in plans.items():
        try:
            stores[label] = _load_store(plan)
        except (FileNotFoundError, SystemExit) as exc:
            log.warning("[compare-retrieval] skip %s: not built (%s)", label, exc)
    return stores
