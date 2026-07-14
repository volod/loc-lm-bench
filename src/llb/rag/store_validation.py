"""Focused store validation implementation."""

from pathlib import Path
from llb.core.config_validation import (
    DEFAULT_EMBEDDING_MODEL,
)
from llb.core.contracts import RagStoreMeta
from llb.prep.corpus_governance import corpus_fingerprint


def store_embedder_mismatch(meta: RagStoreMeta, expected_model: str) -> str | None:
    """Return the store's built embedder id when it differs from `expected_model`, else None.

    A store is embedded and queried by the SAME encoder (recorded in `store_meta.json`), so a
    config that names a different `embedding_model` than the store on disk would silently score
    the wrong encoder. Callers refuse the run with this signal (embedding bake-off fingerprint).
    """
    built = str(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    return built if built != expected_model else None


def stale_store_message(
    meta: RagStoreMeta, corpus_root: Path | str, index_dir: Path | str
) -> str | None:
    """Return a rebuild message when the store fingerprint differs from the current corpus."""
    built = meta.get("corpus_fingerprint")
    if not isinstance(built, str):
        return None
    current = corpus_fingerprint(corpus_root)
    if built == current:
        return None
    return (
        f"[rag] stale store at {index_dir}: corpus manifest fingerprint changed. "
        "Rebuild with `llb build-index --corpus-root <corpus-dir>` so removed sources and "
        "governance metadata propagate into chunks."
    )
