"""Per-study RAG store cache: fingerprint reuse, optional disk persist, embedder prewarm."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from llb.core.config import RunConfig

_LOG = logging.getLogger(__name__)

# Builds a bare RagStore (no fusion / rerank wrap); tests inject fakes that count embeds.
StoreBuilder = Callable[[RunConfig], Any]


def chunking_fingerprint(config: RunConfig) -> tuple[Any, ...]:
    """Chunking / retrieval shape shared across embedder variants of one store family."""
    return (
        config.strategy,
        config.chunk_size,
        config.chunk_overlap,
        config.retrieval_mode,
        config.child_chunk_size,
        config.lexical_lemmas,
    )


def store_fingerprint(config: RunConfig) -> tuple[Any, ...]:
    """Key that forces a rebuild when the embedder or chunking shape changes."""
    return (config.embedding_model, *chunking_fingerprint(config))


def fingerprint_slug(key: tuple[Any, ...]) -> str:
    """Stable short directory name for a store fingerprint (disk cache layout)."""
    payload = json.dumps(list(key), ensure_ascii=False, sort_keys=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def study_stores_dir(data_dir: Path, study_name: str) -> Path:
    """Default disk cache root: ``$DATA_DIR/optuna/<study>/stores/``."""
    return Path(data_dir) / "optuna" / study_name / "stores"


def _build_bare_store(config: RunConfig) -> Any:
    """Chunk + embed without query-time fusion / rerank (those apply on every get)."""
    from llb.rag.store import RagStore

    return RagStore.build(
        config.corpus_root,
        config.strategy,
        config.chunk_size,
        config.chunk_overlap,
        config.embedding_model,
        mode=config.retrieval_mode,
        child_size=config.child_chunk_size,
        lexical_lemmas=config.lexical_lemmas,
    )


def _apply_query_knobs(store: Any, config: RunConfig) -> Any:
    """Stamp fusion knobs and optional reranker wrap from the *current* trial config."""
    from llb.rag.rerank import maybe_wrap_reranker

    store.fusion_weight = config.fusion_weight
    store.fusion_candidates = config.fusion_candidates
    retriever = store
    if config.retrieval_backend == "fused":
        from llb.executor.runner_retrieval import _load_graph_store
        from llb.rag.fusion import FusedRetriever

        retriever = FusedRetriever(store, _load_graph_store(config), config.graph_weight)
    return maybe_wrap_reranker(retriever, config)


@dataclass
class StoreRegistry:
    """Per-study cache: rebuild when the embedder (or chunking) fingerprint changes.

    Optional ``cache_dir`` persists bare stores so a resumed study reloads instead of
    re-embedding. When ``embedders`` is set, the first sight of a chunking fingerprint
    builds every shortlist embedder for that shape (fan-out), and ``prewarm`` does the
    same for the base config before the Optuna loop.
    """

    cache_dir: Path | None = None
    embedders: Sequence[str] | None = None
    builder: StoreBuilder | None = None
    builds: list[tuple[Any, ...]] = field(default_factory=list)
    embed_calls: int = 0
    _cache: dict[tuple[Any, ...], Any] = field(default_factory=dict, repr=False)
    _warmed_chunking: set[tuple[Any, ...]] = field(default_factory=set, repr=False)

    def prewarm(self, base_config: RunConfig, embedders: Sequence[str] | None = None) -> int:
        """Build shortlist stores for ``base_config``'s chunking fingerprint.

        Returns the number of new embed passes performed (0 when already warm / on disk).
        """
        models = list(embedders if embedders is not None else (self.embedders or ()))
        if not models:
            return 0
        before = self.embed_calls
        self.embedders = models
        for model in models:
            cfg = base_config.with_overrides(embedding_model=model)
            self._ensure_bare(cfg)
        self._warmed_chunking.add(chunking_fingerprint(base_config))
        built = self.embed_calls - before
        if built:
            _LOG.info(
                "[tune] prewarmed %d store(s) for chunking fingerprint (%d embed pass(es))",
                len(models),
                built,
            )
        return built

    def get(self, config: RunConfig) -> Any:
        """Return a store for ``config``, applying current fusion / rerank knobs."""
        chunking = chunking_fingerprint(config)
        shortlist = list(self.embedders or ())
        if shortlist and chunking not in self._warmed_chunking:
            # First sight of this chunking shape: build every shortlist embedder once.
            self._warmed_chunking.add(chunking)
            for model in shortlist:
                self._ensure_bare(config.with_overrides(embedding_model=model))
        else:
            self._ensure_bare(config)
        return _apply_query_knobs(self._cache[store_fingerprint(config)], config)

    def _ensure_bare(self, config: RunConfig) -> Any:
        key = store_fingerprint(config)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        loaded = self._load_disk(key) if self.cache_dir is not None else None
        if loaded is not None:
            self._cache[key] = loaded
            return loaded
        self.builds.append(key)
        self.embed_calls += 1
        build = self.builder or _build_bare_store
        store = build(config)
        self._cache[key] = store
        self._save_disk(key, store)
        return store

    def _disk_path(self, key: tuple[Any, ...]) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / fingerprint_slug(key)

    def _save_disk(self, key: tuple[Any, ...], store: Any) -> None:
        if self.cache_dir is None or not hasattr(store, "save"):
            return
        path = self._disk_path(key)
        try:
            store.save(path)
            (path / "fingerprint.json").write_text(
                json.dumps({"key": list(key)}, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:  # pragma: no cover - disk cache is best-effort
            _LOG.debug("[tune] store disk save skipped for %s", fingerprint_slug(key))

    def _load_disk(self, key: tuple[Any, ...]) -> Any | None:
        if self.cache_dir is None:
            return None
        path = self._disk_path(key)
        meta_path = path / "fingerprint.json"
        if not meta_path.is_file():
            return None
        try:
            recorded = json.loads(meta_path.read_text(encoding="utf-8")).get("key")
            if list(recorded) != list(key):
                return None
            from llb.rag.store import RagStore

            store = RagStore.load(path)
            _LOG.info("[tune] reloaded store cache %s", fingerprint_slug(key))
            return store
        except Exception:  # pragma: no cover - corrupt/partial cache -> rebuild
            _LOG.debug("[tune] store disk load failed for %s; rebuilding", fingerprint_slug(key))
            return None
