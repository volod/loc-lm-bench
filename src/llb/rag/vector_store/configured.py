"""Build a vector store with the index settings from a run config."""

from llb.core.config import RunConfig
from llb.rag.vector_store.store import RagStore
from llb.rag.vector_store.vector_index import RAG_BACKEND_FAISS


def build_configured_index(
    config: RunConfig,
    *,
    vector_store: str = RAG_BACKEND_FAISS,
    collapse_duplicates: bool = True,
) -> RagStore:
    return RagStore.build(
        config.corpus_root,
        config.strategy,
        config.chunk_size,
        config.chunk_overlap,
        config.embedding_model,
        mode=config.retrieval_mode,
        child_size=config.child_chunk_size,
        vector_store=vector_store,
        lexical_lemmas=config.lexical_lemmas,
        collapse_duplicates=collapse_duplicates,
        duplicate_tier=config.duplicate_tier,
    )
