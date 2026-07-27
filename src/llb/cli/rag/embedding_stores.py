"""Store builders used by the embedding bake-off command."""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from llb.cli.rag.compare_stores import _dir_size_bytes

if TYPE_CHECKING:
    from llb.core.config import RunConfig
    from llb.prep.frontier_telemetry import ProvenanceLog
    from llb.rag.embedding_bakeoff_models import StoreBuilder


def local_store_builder(cfg: "RunConfig", stores_dir: Path) -> "StoreBuilder":
    """Build and persist one local store per embedding model."""
    from llb.rag.embedding_bakeoff_models import BuiltStore, slugify_model
    from llb.rag.store import RagStore

    def build(model: str) -> "BuiltStore":
        started = time.perf_counter()
        store = RagStore.build(
            cfg.corpus_root,
            cfg.strategy,
            cfg.chunk_size,
            cfg.chunk_overlap,
            model,
            mode=cfg.retrieval_mode,
            child_size=cfg.child_chunk_size,
            lexical_lemmas=cfg.lexical_lemmas,
        )
        embed_seconds = time.perf_counter() - started
        out_dir = stores_dir / slugify_model(model)
        store.save(out_dir)
        resolve = getattr(store.embedder, "_resolve_device", None)
        device = resolve() if callable(resolve) else None
        return BuiltStore(
            store=store,
            embed_seconds=embed_seconds,
            index_bytes=_dir_size_bytes(out_dir),
            device=device,
        )

    return build


def api_store_builder(
    cfg: "RunConfig", stores_dir: Path, log: "ProvenanceLog", max_usd: Optional[float]
) -> "StoreBuilder":
    """Build and persist the API-embedded store while recording its cost."""
    from llb.rag.api_embedder import ApiEmbedder, litellm_embed
    from llb.rag.embedding_bakeoff_models import KIND_API, BuiltStore, slugify_model
    from llb.rag.store import RagStore

    def build(model: str) -> "BuiltStore":
        embedder = ApiEmbedder(model, litellm_embed(model, log=log, max_usd=max_usd))
        started = time.perf_counter()
        store = RagStore.build(
            cfg.corpus_root,
            cfg.strategy,
            cfg.chunk_size,
            cfg.chunk_overlap,
            model,
            mode=cfg.retrieval_mode,
            child_size=cfg.child_chunk_size,
            embedder=embedder,
            lexical_lemmas=cfg.lexical_lemmas,
        )
        embed_seconds = time.perf_counter() - started
        out_dir = stores_dir / slugify_model(model)
        store.save(out_dir)
        return BuiltStore(
            store=store,
            embed_seconds=embed_seconds,
            index_bytes=_dir_size_bytes(out_dir),
            kind=KIND_API,
            cost_usd=log.summary()["total_cost_usd"],
        )

    return build
